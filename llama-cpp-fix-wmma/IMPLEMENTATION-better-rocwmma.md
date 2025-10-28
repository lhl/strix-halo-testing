# Improving rocWMMA FlashAttention on RDNA3

## Current observations
- llama.cpp’s HIP backend enables the legacy WMMA FlashAttention kernel whenever `GGML_HIP_ROCWMMA_FATTN` is set on RDNA3 (`fattn.cu`, `fattn-wmma-f16.cuh`). This code path was originally written for NVIDIA Volta and is explicitly called “old and deprecated.”
- The rocWMMA kernel fixes `nwarps = 4`, uses `__launch_bounds__(nwarps * warp_size, 1)`, and allocates a large static shared-memory tile (`fattn-wmma-f16.cu`). On Strix Halo this limits occupancy to one resident block per CU and forces long serial loops across 256-row K/V chunks.
- When rocWMMA is disabled the backend selects the RDNA-tuned “tile” kernels. Those contain per-architecture macro-configured launch bounds and reuse-friendly chunk sizes, so decode throughput at long context is ~2× higher.
- `docs/build.md` currently recommends enabling `-DGGML_HIP_ROCWMMA_FATTN=ON` for RDNA3 without qualification, which conflicts with the performance behaviour above.

## Code-path anatomy
- **Kernel selection**: `ggml_cuda_get_best_fattn_kernel()` picks WMMA whenever `ggml_cuda_should_use_wmma_fattn()` is true, which on HIP happens iff `GGML_HIP_ROCWMMA_FATTN` is defined and the head size passes statically encoded checks (`fattn.cu`, `fattn-wmma-f16.cuh`).
- **Launch geometry**: `ggml_cuda_flash_attn_ext_wmma_f16_case()` hard-codes `nwarps = 4` and dispatches with `__launch_bounds__(nwarps * warp_size, 1)` (`fattn-wmma-f16.cu`). Combined with the KMEM staging, ROCm reports max-one active block per CU, independent of head size.
- **Shared memory**: the WMMA kernel allocates `__shared__ half KQ[...]` sized as `max(ncols * (FATTN_KQ_STRIDE + 8) * sizeof(KQ_acc_t)/sizeof(half), VKQ_ratio * ncols * (D + 8))`. With `ncols = 16` and `D = 128`, the scratch tops ~35 KB; at `ncols = 32` it roughly doubles.
- **K/V chunk size**: every launch calls `launch_fattn(..., FATTN_KQ_STRIDE, ...)` with `FATTN_KQ_STRIDE = 256`. Decode of a 32K context therefore executes ~125 serial iterations per block. Tile kernels instead derive their batch (`nbatch_fa`, `nbatch_K`) from architecture tables (`GGML_CUDA_FATTN_TILE_CONFIG_CASE`), providing more flexible chunking.
- **Type conversion**: both WMMA and tile paths currently downcast K/V tensors to FP16 via `launch_fattn` before compute. There is no ROCm-specific fast path that keeps int8 in shared memory, so the difference in throughput is primarily launch/occupancy driven.

## Open questions
1. Can we retune the existing WMMA kernel (e.g. higher block residency, smaller scratch buffers, adaptive K/V chunking) while keeping the current template structure?
2. Should RDNA3 prefer the tile kernel for long-context decode and only switch to WMMA for prefill-sized batches?
3. Are there rocWMMA features (e.g. MFMA intrinsics, async copies) we should adopt instead of maintaining the Volta-style implementation?
4. What documentation updates are needed so users benchmark both paths until the WMMA variant is proven faster?

## Next steps
- Catalogue the exact launch configuration and shared-memory footprint for common head sizes (64/128/256) with current WMMA vs tile kernels; record occupancy and shared-mem limits from rocprof/Nsight analogs.
- Capture `llama-bench` decode/prefill runs at 4K/8K/32K context for both WMMA and tile paths to quantify speed deltas and identify crossover points.
- Prototype higher-occupancy WMMA launch parameters (lower `__launch_bounds__`, variable `nwarps`, adaptive chunk size) and validate with microbenchmarks.
- Explore dynamically sizing shared-memory scratch using ROCm occupancy feedback rather than fixed 256-row blocks; confirm compiler still inlines templates cleanly.
- Draft documentation changes (BUILD.md, llama-cpp-cuda-hip.md) that warn RDNA3 users when to enable rocWMMA and point to benchmark toggles.
- Prepare an upstream-friendly change proposal outlining kernels touched, perf data, and required ROCm/rocWMMA versions.

## Candidate implementation directions
- **Retune the existing WMMA kernel**
  - Make `nwarps` and `__launch_bounds__` depend on head size / `cols_per_block`, mirroring the macro tables used by tile kernels.
  - Derive `VKQ_stride` and shared-memory tile dimensions from `ggml_cuda_info().devices[ctx.device].warp_size` and hip occupancy APIs so the compiler sees smaller static footprints.
  - Pros: incremental diff, keeps current rocWMMA entry point, upstreamable.
  - Risks: still limited by Volta-era algorithm (single KQ tile per iteration; heavy synchronizations). Needs extensive profiling to avoid perf regressions on Volta (CUDA builds rely on same source).

- **Add decode-aware kernel selection**
  - Detect long-context decode (`Q->ne[1] == 1`, `S >= threshold`) and prefer tile kernels on RDNA even when rocWMMA is built. Continue to use WMMA for prefill or large batch.
  - Pros: minimal kernel surgery, immediate win for Strix Halo.
  - Risks: adds heuristic branching; must ensure behaviour stays deterministic across HIP/CUDA builds. Upstream acceptance may hinge on measurable perf data.

- **Introduce a ROCm-native WMMA/MFMA variant**
  - Write a dedicated RDNA3 MFMA or rocWMMA kernel that follows ggml’s style (templated `launch_fattn`, macro-configured occupancy) instead of the Volta port.
  - Pros: long-term best performance, aligns with AMD hardware features (wave32, mfma_f16).
  - Risks: large engineering effort; needs thorough validation on CDNA and future RDNA. Must retain CUDA compilation guardrails.

- **Document & tooling updates**
  - Amend `docs/build.md` and `llama-cpp-cuda-hip.md` to recommend benchmarking both paths on RDNA3; note that `GGML_HIP_ROCWMMA_FATTN` may hurt decode throughput today.
  - Provide a `cmake` preset or script toggle that makes it easy to disable WMMA for decode tests (e.g. `-DGGML_HIP_ROCWMMA_FATTN=OFF` override).

## Upstream / maintenance guidelines
- Match the existing macro/table pattern in `fattn-tile.cuh` when introducing new ROCm parameters (`GGML_CUDA_FATTN_TILE_CONFIG_CASE`), so maintainers can reason about occupancy in one place.
- Keep HIP/CUDA code paths unified where practical (the WMMA headers are shared); isolate ROCm-only changes with `#if defined(GGML_USE_HIP)` blocks inside existing templates instead of forking files.
- Maintain architecture capability detection through `common.cuh` helper macros (`GGML_CUDA_CC_IS_RDNA3`, `ggml_cuda_get_physical_warp_size`) instead of new environment variables.
- Ensure every new kernel variant exposes perf counters through `llama-bench` so regressions are catchable in CI; document required ROCm/rocWMMA versions for new instructions.

## Action checklist (pre-dev sync)
- [ ] Benchmark matrix (WMMA vs tile, decode/prefill, multiple contexts) captured and stored under repo docs or bench logs.
- [ ] Occupancy / shared-memory measurements per head size collected for both kernels.
- [ ] Draft BUILD.md and llama-cpp-cuda-hip.md updates prepared to reflect current guidance.
- [ ] Prototype plan decided (retune WMMA vs heuristic gating vs rewrite) with clear owner + success criteria.
