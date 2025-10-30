# Native ROCm FlashAttention (WMMA/MFMA): Opportunity, Design, and TODOs

## Executive Summary
- Goal: replace the legacy Volta-style WMMA FlashAttention with ROCm-native kernels on AMD GPUs: WMMA for RDNA3/RDNA4 and MFMA for CDNA1–3.
- Expected gains
  - RDNA3 prefill (pp): +10–25% end-to-end pp from a 1.3–2.0× faster attention chunk (double-buffered, lean LDS, higher residency).
  - RDNA3 decode (tg): 0–10% at very large S; often bandwidth-bound and already strong with TILE, so smaller ROI here.
  - CDNA2/3 prefill: +10–20% pp from MFMA FA vs legacy WMMA port; decode gains at very large S: +5–20% where attention dominates.
- Scope and safety
  - New kernels are HIP/ROCm only and gated by architecture. CUDA and deprecated Volta paths remain untouched.
  - Fallbacks: existing TILE/VEC remain the safe path whenever shapes aren’t supported or provide better performance.

## Where Performance Is Left on the Table Today
- Volta-era structure constraints the WMMA path:
  - Fixed large KQ stride (256) → long serial loops, inflated LDS → lower residency and less overlap.
  - No double‑buffering for global→LDS → limited ability to hide memory latency.
  - Synchronization-heavy, barrier-rich inner loops; softmax state not streamed.
  - WMMA tile shapes not tailored for RDNA3 fragments nor common head sizes (D = 64/128/256).
  - Static launch bounds; limited tuning by LDS/register pressure.

## Native ROCm Design Overview

### Algorithmic approach (FA‑v2 style)
- Stream K and V tiles through LDS with double-buffering; compute QK and PV while overlapping global→LDS copies.
- Maintain running softmax max/sum per row (FP32) and apply scaling to PV partials in-place.
- Keep per-warp softmax state in registers where possible; minimize LDS reductions and barriers.
- Per‑D/per‑ncols configurations sized to hit 2–3 resident blocks per CU consistently.

### RDNA (WMMA) path (RDNA3, RDNA4)
- Use rocWMMA fragments tuned for wave32, with shapes matched to D in {64, 128, 256} and ncols {8, 16, 32}.
- Vectorized loads (half2), explicit coalescing, bank‑conflict‑free LDS layout.
- Adaptive KQ/VKQ step size (stride) chosen by occupancy model (registers, LDS, waves/CU).
- Launch policy tables per architecture (gfx11.x, gfx12.x): nwarps, nbatch_fa, nbatch_K, cols_per_block.

### CDNA (MFMA) path (CDNA1, CDNA2, CDNA3)
- Use native MFMA FP16/BF16 intrinsics (e.g., __builtin_amdgcn_mfma_*), not WMMA.
- Similar FA‑v2 streaming pipeline; fragment tiling chosen for D in {64, 128, 256} (and DeepSeek 576→512).
- Wave64 considerations (CDNA): adjust reduction patterns and LDS lanes; occupancy tuned per gfx90a/gfx942.

### Precision and numerics
- Inputs in FP16/BF16; accumulations in FP32 for QK, softmax, and PV reductions.
- Optional logit softcap supported without blowing out variants (branch or template bool).
- GQA/ALiBi kept via existing guards; choose ncols2 based on gqa_ratio, with a conservative fallback to ncols2=1 when needed.

### Integration and selection
- New native kernels live beside the existing WMMA/TILE code and are selected only when:
  - Arch supports it (RDNA3/4 for WMMA, CDNA1–3 for MFMA), and
  - D/shape matches supported configs.
- Prefill: prefer native WMMA/MFMA. Decode: keep TILE/VEC unless native wins for large S; always safe‑fallback to TILE.
- If a native config is missing, fall back to TILE; if TILE config is missing, fall back to VEC (guard already present).
- Insert the ROCm-native dispatch ahead of the legacy Volta WMMA selection in `fattn.cu` so CUDA maintains its existing behaviour.
- Drive kernel selection from a HIP capability descriptor (wave size, LDS cap, MFMA/WMMA availability) instead of borrowing CUDA macros such as `turing_mma_available`.

## Coverage Matrix and Fallbacks

- RDNA1 (gfx10.1): no WMMA/MFMA → use TILE/VEC.
- RDNA2 (gfx10.3): no WMMA/MFMA → use TILE/VEC.
- RDNA3 (gfx11.x): WMMA native enabled for pp; decode stays TILE/VEC by default.
- RDNA4 (gfx12.x): WMMA native enabled for pp (rocWMMA ≥ 2.0), decode TILE/VEC by default.
- CDNA1 (gfx908): MFMA available → native MFMA FA (pp), decode TILE/VEC unless clear win.
- CDNA2 (gfx90a): MFMA available → native MFMA FA (pp), decode optional.
- CDNA3 (gfx942): MFMA available → native MFMA FA (pp), decode optional.

## File/Layout Plan (proposed)
- New: `ggml-cuda/fattn-rocm-wmma-f16.cuh` (RDNA WMMA FA kernels; FA‑v2 pipeline)
- New: `ggml-cuda/fattn-mfma-f16.cuh` (CDNA MFMA FA kernels; FA‑v2 pipeline)
- Selection wiring:
  - Update `fattn.cu` to add native ROCm cases before legacy WMMA; gated by arch and shape tables.
  - Keep current WMMA (legacy) and TILE/VEC as fallbacks.

### HIP backend isolation
- Split HIP builds away from the blanket “reuse every CUDA .cu” rule by introducing HIP-specific source lists (native WMMA/MFMA files, TILE fallbacks) compiled from `ggml-hip/`.
- Factor `common.cuh` into CUDA vs HIP headers so HIP defines (`WAVE_SIZE`, async-copy availability, MFMA flags) do not rely on CUDA-specific macros.
- Centralise HIP feature detection in a capability table consumed by MMQ/FA helpers, avoiding checks on CUDA-only predicates (`turing_mma_available`, `GGML_CUDA_CC_*`).
- Update build documentation and CI to exercise HIP with native kernels enabled/disabled, catching regressions like the rocWMMA slowdown noted this cycle.

## Tuning knobs and tables
- Per‑arch config table entries keyed by (DKQ, DV, ncols):
  - threads/nwarps per block; occupancy target (blocks/CU)
  - nbatch_fa, nbatch_K (stream‑K chunking)
  - cols_per_block, ncols2 choices per gqa_ratio
  - LDS footprint (bytes), shared buffers and double‑buffering depth

## Testing and Validation Plan
- Correctness: compare outputs vs TILE baseline across a grid:
  - D ∈ {64, 128, 256, 576→512}, S ∈ {1k, 4k, 8k, 32k}, gqa_ratio ∈ {1, 2, 4, 8, 16}, with/without ALiBi/mask
  - Logit softcap on/off; FP16 vs BF16 models
- Performance: `llama-bench` pp/tg across depths on RDNA3 (gfx1151), RDNA4 (gfx12), CDNA2 (gfx90a), CDNA3 (gfx942)
  - Collect waves/CU, occupancy, LDS bytes, global bytes, and kernel durations (rocprof/omniperf)
- Stability: soak tests on long prompts, verify no device traps, ensure guard fallbacks engage correctly when configs absent.

## Risks and Mitigations
- Variant explosion: limit D to 64/128/256 (+ DeepSeek special); constrain ncols set to {8,16,32} and ncols2 {1,2,4,8,16} with pruning at compile‑time.
- LDS pressure/reg pressure reducing occupancy: use occupancy model to choose stride and double‑buffering depth; validate with counters.
- Divergence between RDNA and CDNA paths: keep a shared FA‑common helper (softmax streaming, LDS layout helpers) to reduce duplication.
- Upstream acceptance: changes are gated, optional, and preserve fallbacks; include comprehensive perf data and toggles to disable if needed.

## Deliverables and Milestones
- M1: RDNA3 WMMA prefill native kernel (D=128/256), selection and fallbacks, perf validation
- M2: RDNA3 WMMA add D=64, finalize tuning tables, documentation
- M3: CDNA2 MFMA prefill native kernel (D=128/256), selection and fallbacks, perf validation
- M4: CDNA1/CDNA3 MFMA expansion; DeepSeek 576→512 support; BF16 coverage
- M5: Optional decode native prototype for very large S; keep TILE default unless >5–10% win demonstrated

## Implementation Sketch (RDNA3 WMMA, D=128)
- Kernel structure
  - Thread block handles cols_per_block ∈ {16, 32}; nwarps chosen to sustain 2–3 blocks/CU.
  - Double-buffer LDS for K and V tiles: while computing on tile t, prefetch tile t+1.
  - Use rocWMMA fragments for A/B/C with row/col-major layouts matching LDS packing.
  - Maintain KQ_max/KQ_sum in FP32 per output row in registers; reduce across warps at tile boundaries.
  - Scale and accumulate PV partials as softmax sum is updated to avoid storing full attention probs.
- Memory
  - Global loads: coalesced half2/float2 into LDS; explicit alignment; avoid bank conflicts with padding.
  - LDS footprint minimized by smaller KQ stride (e.g., 64–128 rows per tile, not fixed 256).
- Launch
  - Set __launch_bounds__(threads_per_block, min_blocks_per_SM=2) on HIP; tune for actual LDS usage.
  - Use per‑arch tables to choose (nwarps, nbatch_fa, nbatch_K, cols_per_block) at runtime.
- Integration
  - New `ggml_cuda_flash_attn_ext_rocm_wmma_f16_case<DKQ,DV,cols_per_block>()` wrappers and selection hooks.

## Implementation Sketch (CDNA MFMA, D=128)
- Kernel structure mirrors RDNA design but uses MFMA intrinsics for matmul.
- Wave64 reductions and register usage tuned for CDNA; per‑arch tables (gfx908/gfx90a/gfx942).
- BF16 variant planned; accumulations stay FP32.

## TODO Checklist (production-ready scope)
- [ ] Feature detection and compile guards for RDNA3/RDNA4 WMMA and CDNA MFMA
- [ ] Common FA utilities: softmax streaming helpers, LDS layout/padding helpers, vectorized copy utilities
- [ ] RDNA WMMA native kernels: D ∈ {64, 128, 256}; ncols ∈ {8,16,32}; ncols2 ∈ {1,2,4,8,16}
- [ ] CDNA MFMA native kernels: D ∈ {64, 128, 256}; special DeepSeek 576→512
- [ ] Logit softcap support (build-time or runtime flag), minimal variant branching
- [ ] GQA/ALiBi support and guards; correct ncols2 mapping by gqa_ratio
- [ ] Launch parameter tables per arch for occupancy: nwarps, nbatch_fa, nbatch_K, cols_per_block
- [ ] Selection logic integration, with fallbacks to TILE/VEC on unsupported shapes
- [ ] HIP capability table + dispatch refactor (stop depending on CUDA-only macros)
- [ ] Split HIP build sources/CMake from CUDA path; dedicate HIP `common` header
- [ ] CI/build coverage for HIP native kernels (native on/off) to catch regressions
- [ ] Template instantiation limits and pruning to control compile time/binary size
- [ ] Correctness tests vs TILE across a grid of (D, S, gqa_ratio, softcap, bf16/fp16)
- [ ] Performance validation on RDNA3, RDNA4, CDNA2, CDNA3; capture counters and tokens/s
- [ ] Documentation: build flags, architecture requirements, troubleshooting, toggles to disable native path
- [ ] Upstream PR preparation: scoped commits, benchmarks, before/after tables, guidance for maintainers

## Notes on Decode
- Decode (tg) at small/medium S remains weight‑GEMM dominated; attention gains have limited effect on end‑to‑end tokens/s.
- A native decode kernel can be explored for S ≫ (e.g., ≥8k), but default should continue to use TILE unless a consistent +5–10% end‑to‑end win is demonstrated on supported GPUs.

## Appendix: References and Counters to Watch
- rocprof/Omniperf: waves/CU, active blocks/CU, LDS usage, global bytes, VALU vs matrix utilization, barrier count.
- L2/L1 hit rates when double‑buffering is active.
- Register usage per thread and spill counts.
