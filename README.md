# llama.cpp-sm_61

This is [Dimonyga/llama.cpp-sm_61](https://github.com/Dimonyga/llama.cpp-sm_61), a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) focused on NVIDIA Pascal **sm_61** GPUs and **Vulkan on older AMD Radeon cards**, especially GCN/Vega. It follows upstream and carries targeted performance changes plus selected server, audio and GGML features. Some optimizations are adapted from [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp).

## What this fork adds

- **Pascal / sm_61:** DP4A matrix-vector and matrix-matrix tuning, including IQ4_XS; four-row MMVQ dispatch; a GQA-aware single-token Flash Attention path for Q4_0 K/V caches; and changes to GDN and tiled attention for supported shapes. These paths target GPUs such as the GTX 1080 Ti and use architecture and layout checks to select compatible kernels.
- **Older Radeon / Vulkan:** GCN Q4_0 KV-cache dequantization with transpose, tiled transpose-concat for GDN, and IQ4_XS matrix-vector shader changes. Subgroup RMS normalization and smaller workgroups, initially backported here, are now also part of the merged upstream code. Build options can omit cooperative-matrix shaders on devices that do not support them.
- **CPU inference:** tiled Flash Attention for Q4_0 and Q8_0 caches, including mixed K/V types, and split-SwiGLU graph fusion in the Qwen3.5 and Qwen3-Next paths.
- **Audio API:** Gemma 4 transcription and translation prompts, `/v1/audio/translations` with `target_language`, text/JSON/subtitle response formats, an FFmpeg decoding fallback, and `/v1/audio/speech` as an alias of `/tts`. Qwen3-TTS VoiceDesign accepts voice descriptions through `instruct` or `instructions`. See the [server documentation](tools/server/README.md#multimodal-support) and [TTS documentation](tools/tts/README.md).
- **Server statistics:** `--moe-routing-log` records expert-selection counts; `--tensor-stats` collects activation statistics from served traffic as a GGUF importance matrix for quantization. Both are opt-in and add synchronization overhead.
- **GGML extensions for stable-diffusion.cpp integration:** FP8 conversion paths, INT8 matrix operations with ConvRot preprocessing, RoPE application, and CUDA SageAttention/SOL attention operators. Support depends on the backend and GPU; SageAttention and SOL attention require newer NVIDIA architectures and are separate from the Pascal optimizations.

Optimizations apply to specific devices, quantization types and tensor layouts. Performance depends on the model, workload, driver and build configuration. The presence of a kernel is not a claim that every supported GPU or model has been benchmarked.

## Build this fork

Clone this repository and build from source to include its changes. The upstream installers, binaries, container images and release badges in the inherited documentation below refer to upstream builds.

```sh
git clone https://github.com/Dimonyga/llama.cpp-sm_61.git
cd llama.cpp-sm_61
```

For NVIDIA Pascal sm_61, use a CUDA toolkit that supports compiling for compute capability 6.1:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cuda --parallel
```

For older Radeon cards using Vulkan:

```sh
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_METAL=OFF \
    -DGGML_VULKAN_NO_COOPMAT=ON -DGGML_VULKAN_NO_COOPMAT2=ON \
    -DLLAMA_BUILD_TESTS=ON
cmake --build build-vulkan --parallel
```

The two `NO_COOPMAT` options omit cooperative-matrix shaders for a build targeting devices without those features. Leave them off for builds that should also use cooperative matrices on newer GPUs. See [docs/build.md](docs/build.md) for dependencies and platform-specific setup.

## Source archive for a build host

Create `llama.cpp-YYYYMMDDHHmm.tar.gz` in the repository root, or pass an output directory:

```sh
python3 scripts/package_sources.py
python3 scripts/package_sources.py /path/to/output
```

The script needs Python 3 and Git on the packaging machine. It includes current working-tree files from the native source directories, including local edits and new non-ignored source files, all GGML backends, bundled dependencies, licenses, tools and chat templates. It excludes Web UI sources and assets, examples, tests and benchmark suites, documentation directories, CI files, model weights, Git history and Git-ignored build outputs. Native tools such as `llama-bench` remain included. Three small UI build files remain to generate the server's empty asset table. An existing archive with the same name is not overwritten.

Extract on the build host and load the included `source-build.cmake` initial cache. It disables both UI building and prebuilt UI downloads, examples and tests, and preserves the source revision in the build information. Add the backend flags needed for the host, for example Pascal CUDA:

```sh
tar -xzf llama.cpp-YYYYMMDDHHmm.tar.gz
cmake -S llama.cpp -B build -C llama.cpp/source-build.cmake \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DGGML_VULKAN=OFF -DGGML_METAL=OFF
cmake --build build --parallel
```

The original upstream README follows.

---

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
