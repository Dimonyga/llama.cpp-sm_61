# Working on llama.cpp-sm_61

## Repository and scope

This is [Dimonyga/llama.cpp-sm_61](https://github.com/Dimonyga/llama.cpp-sm_61), a maintained fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Work in this repository targets the fork unless the user explicitly requests an upstream contribution.

The priorities are:

- NVIDIA Pascal GPUs with compute capability 6.1, including GTX 1080 Ti: DP4A matrix kernels, quantized inference, Flash Attention and GDN.
- Older AMD Radeon GPUs using Vulkan, especially GCN/Vega: efficient shaders and dispatch for devices without cooperative matrices.
- CPU quantized attention and selected server, audio and GGML extensions described in [README.md](README.md).

AI-assisted implementation, documentation, review notes and commit messages are allowed in this fork. Understand the changes, keep them reviewable and validate them. The upstream contribution and submission policies in [CONTRIBUTING.md](CONTRIBUTING.md) apply to contributions sent to ggml-org/llama.cpp; they do not impose upstream submission approval steps on work in this fork. Read the current upstream policies before preparing an explicitly requested upstream contribution.

## Start with the current tree

- Read `git status --short --branch`, inspect the diff and check `git remote -v` before changing files or synchronizing branches.
- Preserve the user's edits, staged changes and deletions. Do not restore removed documentation or overwrite unrelated work.
- Read the relevant implementation and nearby tests before editing. Follow the existing backend and server patterns.
- Use the code and current Git history to establish what is implemented. Old notes and benchmark results may describe a different revision.

## Implementation

- Keep changes focused. Reuse existing operators, helpers, dispatch and test infrastructure.
- Use the surrounding C/C++ style: 4 spaces, `snake_case`, sized integer types where appropriate, `void * ptr` and `int & ref`.
- Keep code and comments ASCII. Comments should explain a non-obvious constraint in one or two lines. Do not hard-wrap prose to a fixed column width.
- Gate optimizations by actual device capabilities, tensor types, shapes and layouts. Keep a correct fallback for unsupported cases.
- Pascal sm_61 has DP4A but no Tensor Cores. Do not route it into kernels that require newer CUDA architectures.
- Older Radeon devices differ in subgroup size, FP16 support, integer dot products and drivers. Check capabilities instead of assuming all AMD GPUs behave alike.
- Preserve GGUF type identifiers and check operator tables, traits, buffer sizes and backend support checks when changing GGML types or operations.
- Treat model metadata, tensor dimensions, token IDs and HTTP input as untrusted. Check bounds, arithmetic overflow, ownership and error handling.
- Keep license notices and attribution for adapted kernels, including code from ik_llama.cpp and other projects.
- For changes in an area covered by [skills/code-review/SKILL.md](skills/code-review/SKILL.md), use its applicable technical checks. Its upstream submission rules apply only when submitting upstream.

## Validation

Use separate build directories for CPU, CUDA and Vulkan. See the fork build examples in [README.md](README.md) and the dependency details in [docs/build.md](docs/build.md).

- Documentation-only edits: check the diff, local links, commands and `git diff --check`; a rebuild is unnecessary.
- Backend edits: build the affected backend and run focused `test-backend-ops` cases against the CPU reference. Add regression cases to the existing test file when needed.
- CUDA sm_61 edits: compile with `-DCMAKE_CUDA_ARCHITECTURES=61` and run on a real Pascal device before claiming correctness or speed on that target.
- Vulkan edits: run on the relevant Radeon device and driver. Confirm the backend name with `--list-devices` before selecting it for tests or benchmarks.
- Server edits: build `llama-server`, run the relevant component tests and perform a focused HTTP smoke test. Audio or model behavior needs a request using a compatible real model.
- Performance claims: compare the same model, tensor types, workload and settings before and after. Record the revision, hardware, driver, build flags and measurements.
- Report compilation, operator tests, real inference and benchmarks separately. State what could not be tested; syntax checks and CPU tests do not establish GPU correctness.

Examples after building the test targets:

```sh
build-cuda/bin/test-backend-ops test -b CUDA0 -o 'MUL_MAT,FLASH_ATTN_EXT,GATED_DELTA_NET'
build-vulkan/bin/test-backend-ops test -b Vulkan0 -o '.*RMS_NORM.*,CONCAT,FLASH_ATTN_EXT'
ctest --test-dir build-cpu --output-on-failure -R '^test-(server-component|chat|quantize-fns)$'
git diff --check
```

Replace backend names with those reported on the test machine. Keep tests proportional to the change and do not download large models or disrupt running services without task-specific authorization.

## Upstream synchronization

- Verify the remotes: the fork is Dimonyga/llama.cpp-sm_61; upstream is ggml-org/llama.cpp. A local remote can still use the old Dimonyga/llama.cpp URL, which GitHub redirects.
- Fetch upstream and inspect divergence before merging. Preserve fork commits; do not replace the branch with upstream or rewrite published history without an explicit request.
- Review automatic merges as well as conflict resolutions. Upstream may introduce a second implementation of a feature already present here, especially types, conversion paths, shaders and audio routes.
- Check architecture guards, type numbering, dispatch, API fields and tests after a merge. Preserve the fork introduction in README and these fork-specific instructions.

## Commits and publishing

- Commit or push when the user requests it. An explicit request authorizes the corresponding action in this fork; do not add an upstream contributor-ban acknowledgment requirement to an ordinary fork push.
- Use concise commit messages. Add `Assisted-by: Codex` when Codex meaningfully contributed; reserve `Co-authored-by` for human co-authors.
- Check the destination before pushing. Send fork work to the user's fork. Publishing to upstream or opening a PR requires an explicit request for that destination.
- Do not force-push, discard work or rewrite shared history without explicit authorization.
- After a push, verify the remote branch SHA and report the commit, destination, checks and remaining limitations.
