// INT8 flash attention for GPUs that have dp4a but no fast FP16 or tensor cores (Pascal, cc 6.1).
//
// The QK^T half of attention runs on __dp4a with per-token int8 scales (K is smoothed by subtracting its
// per-head channel mean, which leaves the softmax unchanged); the PV half stays in FP32. Targets
// unmasked, non-GQA attention with head size 128 and F16 K/V, i.e. the image diffusion transformers.

#include "fattn-dp4a.cuh"

namespace {

constexpr int DP4A_D  = 128;
constexpr int DP4A_BQ = 64;
constexpr int DP4A_BK = 32;
constexpr int DP4A_NT = 128;
constexpr int DP4A_QS = 144; // int8 smem row stride in bytes, 16 bytes of padding against bank conflicts
constexpr int DP4A_PS = 68;  // fp32 smem row strides in elements
constexpr int DP4A_VS = 132;

struct fattn_dp4a_strides {
    int64_t q1, q2, q3; // bytes
    int64_t k1, k2, k3;
    int64_t v1, v2, v3;
};

static __device__ __forceinline__ float fattn_dp4a_block_amax(float v, float * sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    }
    if ((threadIdx.x & 31) == 0) {
        sh[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    const float r = fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3]));
    __syncthreads();
    return r;
}

static __global__ void fattn_dp4a_k_mean(const char * K, float * kmean, const int n_kv, const int n_head, const fattn_dp4a_strides s) {
    const int h = blockIdx.x % n_head, b = blockIdx.x / n_head, d = threadIdx.x;
    const char * base = K + b*s.k3 + h*s.k2;
    float sum = 0.0f;
    for (int i = 0; i < n_kv; ++i) {
        sum += __half2float(*(const half *) (base + i*s.k1 + 2*d));
    }
    kmean[blockIdx.x*DP4A_D + d] = sum/n_kv;
}

static __global__ void fattn_dp4a_quant_k(const char * K, const float * kmean, int8_t * Kq, float * Ks,
        const int n_kv, const int n_kv_pad, const int n_head, const fattn_dp4a_strides s) {
    __shared__ float sh[4];
    const int i = blockIdx.x, h = blockIdx.y % n_head, b = blockIdx.y / n_head, d = threadIdx.x;
    const int64_t row = (int64_t) blockIdx.y*n_kv_pad + i;
    float v = 0.0f;
    if (i < n_kv) {
        v = __half2float(*(const half *) (K + b*s.k3 + h*s.k2 + i*s.k1 + 2*d)) - kmean[blockIdx.y*DP4A_D + d];
    }
    const float amax = fattn_dp4a_block_amax(fabsf(v), sh);
    const float sc   = amax > 0.0f ? amax/127.0f : 1.0f;
    Kq[row*DP4A_D + d] = (int8_t) __float2int_rn(v/sc);
    if (d == 0) {
        Ks[row] = sc;
    }
}

static __global__ void fattn_dp4a_quant_q(const char * Q, int8_t * Qq, float * Qs,
        const int n_q, const int n_q_pad, const int n_head, const float fac, const fattn_dp4a_strides s) {
    __shared__ float sh[4];
    const int i = blockIdx.x, h = blockIdx.y % n_head, b = blockIdx.y / n_head, d = threadIdx.x;
    const int64_t row = (int64_t) blockIdx.y*n_q_pad + i;
    float v = 0.0f;
    if (i < n_q) {
        v = *(const float *) (Q + b*s.q3 + h*s.q2 + i*s.q1 + 4*d)*fac;
    }
    const float amax = fattn_dp4a_block_amax(fabsf(v), sh);
    const float sc   = amax > 0.0f ? amax/127.0f : 1.0f;
    Qq[row*DP4A_D + d] = (int8_t) __float2int_rn(v/sc);
    if (d == 0) {
        Qs[row] = sc;
    }
}

static __global__ __launch_bounds__(DP4A_NT, 2) void fattn_dp4a_fwd(
        const int8_t * Qq, const float * Qs, const int8_t * Kq, const float * Ks, const char * V, float * dst,
        const int n_q, const int n_q_pad, const int n_kv, const int n_kv_pad, const int n_head, const fattn_dp4a_strides s) {
    __shared__ __align__(16) int8_t sQ[DP4A_BQ*DP4A_QS];
    __shared__ __align__(16) int8_t sK[DP4A_BK*DP4A_QS];
    __shared__ float sQs[DP4A_BQ], sKs[DP4A_BK];
    __shared__ __align__(16) float sP[DP4A_BK*DP4A_PS];
    __shared__ __align__(16) float sV[DP4A_BK*DP4A_VS];

    const int tid = threadIdx.x, tx = tid & 7, ty = tid >> 3;
    const int qb = blockIdx.x*DP4A_BQ;
    const int h = blockIdx.y % n_head, b = blockIdx.y / n_head;
    const int64_t seq = blockIdx.y;

    for (int c = tid; c < DP4A_BQ*8; c += DP4A_NT) {
        const int row = c >> 3, ch = c & 7;
        *(int4 *) &sQ[row*DP4A_QS + ch*16] = *(const int4 *) (Qq + (seq*n_q_pad + qb + row)*DP4A_D + ch*16);
    }
    if (tid < DP4A_BQ) {
        sQs[tid] = Qs[seq*n_q_pad + qb + tid];
    }

    float m[4], l[4], O[4][16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            O[i][c] = 0.0f;
        }
    }

    const char * Vh = V + b*s.v3 + h*s.v2;

    for (int kt = 0; kt < n_kv; kt += DP4A_BK) {
        __syncthreads();
        for (int c = tid; c < DP4A_BK*8; c += DP4A_NT) {
            const int row = c >> 3, ch = c & 7;
            *(int4 *) &sK[row*DP4A_QS + ch*16] = *(const int4 *) (Kq + (seq*n_kv_pad + kt + row)*DP4A_D + ch*16);
        }
        if (tid < DP4A_BK) {
            sKs[tid] = Ks[seq*n_kv_pad + kt + tid];
        }
        for (int c = tid; c < DP4A_BK*16; c += DP4A_NT) {
            const int row = c >> 4, ch = c & 15;
            int4 raw = make_int4(0, 0, 0, 0);
            if (kt + row < n_kv) {
                raw = *(const int4 *) (Vh + (int64_t) (kt + row)*s.v1 + ch*16);
            }
            const half2 * hp = (const half2 *) &raw;
            const float2 f0 = __half22float2(hp[0]), f1 = __half22float2(hp[1]), f2 = __half22float2(hp[2]), f3 = __half22float2(hp[3]);
            *(float4 *) &sV[row*DP4A_VS + ch*8]     = make_float4(f0.x, f0.y, f1.x, f1.y);
            *(float4 *) &sV[row*DP4A_VS + ch*8 + 4] = make_float4(f2.x, f2.y, f3.x, f3.y);
        }
        __syncthreads();

        int acc[4][4] = {};
#pragma unroll
        for (int kw = 0; kw < 8; ++kw) {
            int4 qv[4], kv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                qv[i] = *(const int4 *) &sQ[(ty*4 + i)*DP4A_QS + kw*16];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                kv[j] = *(const int4 *) &sK[(j*8 + tx)*DP4A_QS + kw*16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int a = acc[i][j];
                    a = __dp4a(qv[i].x, kv[j].x, a);
                    a = __dp4a(qv[i].y, kv[j].y, a);
                    a = __dp4a(qv[i].z, kv[j].z, a);
                    a = __dp4a(qv[i].w, kv[j].w, a);
                    acc[i][j] = a;
                }
            }
        }

        float p[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float qs = sQs[ty*4 + i];
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p[i][j] = kt + j*8 + tx < n_kv ? (float) acc[i][j]*qs*sKs[j*8 + tx] : -INFINITY;
                mx = fmaxf(mx, p[i][j]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
            const float mn    = fmaxf(m[i], mx);
            const float alpha = exp2f(m[i] - mn);
            float ls = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p[i][j] = exp2f(p[i][j] - mn);
                ls += p[i][j];
            }
            l[i] = l[i]*alpha + ls;
            m[i] = mn;
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                O[i][c] *= alpha;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            *(float4 *) &sP[(j*8 + tx)*DP4A_PS + ty*4] = make_float4(p[0][j], p[1][j], p[2][j], p[3][j]);
        }
        __syncthreads();

#pragma unroll 4
        for (int k = 0; k < DP4A_BK; ++k) {
            const float4 pv = *(const float4 *) &sP[k*DP4A_PS + ty*4];
            const float pq[4] = {pv.x, pv.y, pv.z, pv.w};
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float4 vv = *(const float4 *) &sV[k*DP4A_VS + jj*32 + tx*4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    O[i][jj*4 + 0] += pq[i]*vv.x;
                    O[i][jj*4 + 1] += pq[i]*vv.y;
                    O[i][jj*4 + 2] += pq[i]*vv.z;
                    O[i][jj*4 + 3] += pq[i]*vv.w;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float lt = l[i];
        lt += __shfl_xor_sync(0xffffffff, lt, 1);
        lt += __shfl_xor_sync(0xffffffff, lt, 2);
        lt += __shfl_xor_sync(0xffffffff, lt, 4);
        const float inv = 1.0f/lt;
        const int q = qb + ty*4 + i;
        if (q < n_q) {
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                *(float4 *) &dst[DP4A_D*(h + (int64_t) n_head*(q + (int64_t) n_q*b)) + jj*32 + tx*4] =
                    make_float4(O[i][jj*4]*inv, O[i][jj*4 + 1]*inv, O[i][jj*4 + 2]*inv, O[i][jj*4 + 3]*inv);
            }
        }
    }
}

} // namespace

bool ggml_cuda_fattn_dp4a_supported(int device, const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_FATTN_DP4A");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_DP4A || fast_fp16_available(cc) || turing_mma_available(cc)) {
        return false;
    }

    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (mask != nullptr || dst->src[4] != nullptr || max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != DP4A_D || K->ne[0] != DP4A_D || V->ne[0] != DP4A_D) {
        return false;
    }
    if (Q->ne[2] != K->ne[2] || K->ne[2] != V->ne[2] || Q->ne[3] != K->ne[3] || K->ne[1] != V->ne[1]) {
        return false;
    }
    if (Q->ne[1] < 128 || K->ne[1] < 128) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half)) {
        return false;
    }
    for (int i = 1; i < 4; ++i) {
        if (V->nb[i] % 16 != 0) {
            return false;
        }
    }
    return reinterpret_cast<uintptr_t>(V->data) % 16 == 0;
}

void ggml_cuda_flash_attn_ext_dp4a(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int n_q    = Q->ne[1];
    const int n_kv   = K->ne[1];
    const int n_head = Q->ne[2];
    const int n_seq  = n_head*Q->ne[3];
    const int n_q_pad  = GGML_PAD(n_q,  DP4A_BQ);
    const int n_kv_pad = GGML_PAD(n_kv, DP4A_BK);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const fattn_dp4a_strides s = {
        (int64_t) Q->nb[1], (int64_t) Q->nb[2], (int64_t) Q->nb[3],
        (int64_t) K->nb[1], (int64_t) K->nb[2], (int64_t) K->nb[3],
        (int64_t) V->nb[1], (int64_t) V->nb[2], (int64_t) V->nb[3],
    };

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int8_t> Qq(pool, (size_t) n_seq*n_q_pad*DP4A_D);
    ggml_cuda_pool_alloc<int8_t> Kq(pool, (size_t) n_seq*n_kv_pad*DP4A_D);
    ggml_cuda_pool_alloc<float>  Qs(pool, (size_t) n_seq*n_q_pad);
    ggml_cuda_pool_alloc<float>  Ks(pool, (size_t) n_seq*n_kv_pad);
    ggml_cuda_pool_alloc<float>  kmean(pool, (size_t) n_seq*DP4A_D);

    cudaStream_t stream = ctx.stream();
    fattn_dp4a_k_mean<<<n_seq, DP4A_D, 0, stream>>>((const char *) K->data, kmean.get(), n_kv, n_head, s);
    fattn_dp4a_quant_k<<<dim3(n_kv_pad, n_seq), DP4A_D, 0, stream>>>((const char *) K->data, kmean.get(), Kq.get(), Ks.get(), n_kv, n_kv_pad, n_head, s);
    // softmax is evaluated with exp2, so fold log2(e) into the Q scale
    fattn_dp4a_quant_q<<<dim3(n_q_pad, n_seq), DP4A_D, 0, stream>>>((const char *) Q->data, Qq.get(), Qs.get(), n_q, n_q_pad, n_head, scale*1.44269504f, s);
    fattn_dp4a_fwd<<<dim3(n_q_pad/DP4A_BQ, n_seq), DP4A_NT, 0, stream>>>(
        Qq.get(), Qs.get(), Kq.get(), Ks.get(), (const char *) V->data, (float *) dst->data, n_q, n_q_pad, n_kv, n_kv_pad, n_head, s);
    CUDA_CHECK(cudaGetLastError());
}
