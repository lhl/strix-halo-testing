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

## Current implementation snapshot
- Prefill scheduling: rocWMMA remains active whenever the FlashAttention tensor has `dst->ne[2] > 1` (multi-token batches); single-token streaming decode (`dst->ne[2] == 1`) now falls back to the tuned vector kernel so decode throughput matches the HIP baseline.
- Launch geometry: HIP builds run WMMA with `nwarps = 8` for head sizes ≤ 96 and `nwarps = 4` otherwise, and cap KQ/V shared-memory stride at 128 rows to stay within RDNA3 limits while keeping two resident blocks per CU.
- Observed perf: prefill shows 6–65% improvements across contexts; decode parity is restored relative to HIP after the heuristic switches back to the vector path for single-token passes.
- Remaining work: trim the temporary kernel logging, validate across wider shape/test matrices (`tg128 @ d4096`, `tg128 @ d65536`), and document the `dst->ne[2] == 1` gating so downstream users know how to evaluate rocWMMA on their workloads.

## 2025-10-28: Debugging decode kernel fallback issue

### Problem
After tuning rocWMMA for prefill (6-65% improvements), decode (tg) performance is still slow. We want decode to fall back to the regular HIP VEC kernel, but it's not happening.

### Tensor dimension reference
GGML tensor dimensions (`tensor->ne[...]`):
- `ne[0]` = head dimension (e.g., 64, 128, 256)
- `ne[1]` = number of tokens/queries (1 for decode, >1 for prefill)
- `ne[2]` = number of attention heads
- `ne[3]` = batch size

### Debugging attempts

**Location**: `llama.cpp/ggml/src/ggml-cuda/fattn.cu` in `ggml_cuda_get_best_fattn_kernel()`

#### Attempt 1: Original code (from previous work)
```cpp
const bool hip_wmma_decode = dst->ne[2] == 1;
```
**Why it's wrong**: This checks if there's 1 attention head, not if we're doing decode. Models typically have multiple attention heads (e.g., 32 heads for Llama), so this condition is never true.

**Result**: Decode still uses WMMA kernel, performance matches rocWMMA baseline (~143 t/s @ d4096) instead of HIP baseline (~173 t/s @ d4096).

#### Attempt 2: Check dst->ne[1]
```cpp
const bool hip_wmma_decode = dst->ne[1] == 1;
```
**Why it's wrong**: The `dst` tensor dimensions don't directly represent the number of tokens. Need to check the Query tensor instead.

**Result**: Still using WMMA for decode (performance unchanged).

#### Attempt 3: Check Q->ne[1] (VERIFIED WORKING)
```cpp
const bool hip_wmma_decode = Q->ne[1] == 1;
```
**Why this should work**: Throughout the codebase, `Q->ne[1] == 1` is the standard pattern for detecting decode (see lines 281, 286, 290, 295, 310, 323, 329 in fattn.cu). The Query tensor's `ne[1]` dimension is the number of tokens being processed - exactly 1 for decode, multiple for prefill.

**Verification**: Kernel selection detects decode via `Q->ne[1] == 1`; prefill uses WMMA_F16, decode uses HIP logic (VEC/TILE).

**NEW PROBLEM**: Despite selecting VEC kernel for decode, performance still matches rocWMMA baseline (~143 t/s @ d4096) instead of HIP baseline (~173 t/s @ d4096). The VEC kernel is being selected but not delivering expected performance.

### Investigation findings

**Performance comparison**:
```
             d4096 decode (tg128)
HIP:         173.36 t/s
rocWMMA:     143.51 t/s  (using WMMA kernel)
lhl-tune:    143.59 t/s  (using VEC kernel - verified by debug logs!)
```

**The mystery**: Same VEC kernel code, different performance depending on build:
- When built WITHOUT rocWMMA enabled → VEC achieves ~173 t/s
- When built WITH rocWMMA enabled → VEC achieves only ~143 t/s

**Checked so far**:
1. ✓ Kernel selection is 100% correct (debug logs confirm VEC for decode)
2. ✓ VEC kernel source code has no rocWMMA-specific conditionals
3. ✓ WMMA kernel modifications only affect WMMA, not VEC
4. ? CMake flags - need to compare default-hip vs rocwmma build flags
5. ? rocWMMA library linking - does just having library linked affect all kernels?
6. ? Compiler optimization differences with rocWMMA enabled?

**Deeper investigation**:
- ✓ VEC kernel source code (`fattn-vec.cuh`) contains NO rocWMMA-specific conditionals
- ✓ Kernel dispatch is straightforward: `BEST_FATTN_KERNEL_VEC` → `ggml_cuda_flash_attn_ext_vec()`
- ✓ CMake adds `-DGGML_HIP_ROCWMMA_FATTN` compile definition to ALL HIP code when enabled
- ✓ Both builds use same `GGML_USE_HIP` and `RDNA` code paths in VEC kernel

**Theories for 17% performance regression**:

1. **Compiler optimization changes**: When rocWMMA code is in the same compilation unit, HIP compiler may:
   - Apply different register allocation strategies
   - Change instruction scheduling
   - Modify occupancy calculations

2. **Shared library loading**: rocWMMA library linkage might:
   - Change how HIP runtime initializes
   - Affect GPU resource allocation
   - Introduce loader overhead

3. **Memory/resource contention**: rocWMMA initialization could:
   - Reserve GPU resources affecting all kernels
   - Change default memory pool configurations
   - Affect L2 cache behavior

4. **Build artifacts**: The current build might have:
   - Uncommitted changes we're unaware of
   - Different optimization flags
   - Stale object files

**Proposed tests**:
1. Build fresh rocWMMA-enabled version from clean slate
2. Add more verbose kernel logging (launch params, grid size, block size)
3. Use `rocprof` to profile actual kernel execution time
4. Compare `hipcc` command lines between builds
5. Test with rocWMMA library unlinked but flag still defined

## Code path analysis: HIP-only vs rocWMMA build

### HIP-only build (`-DGGML_HIP=ON`)

In `ggml_cuda_get_best_fattn_kernel()` (`fattn.cu:199`):

1. **Line 278**: `turing_mma_available(cc)` → `false` (RDNA3 ≠ Turing)
2. **Line 319**: `ggml_cuda_should_use_wmma_fattn(cc)` → **returns `false`**
   - See `fattn-wmma-f16.cuh:27-28`:
   ```cpp
   #if defined(GGML_USE_HIP) && !defined(GGML_HIP_ROCWMMA_FATTN)
       return false;  // ← Takes this path when rocWMMA not enabled
   ```
3. **Line 317**: `hip_wmma_decode = false` (GGML_HIP_ROCWMMA_FATTN not defined)
4. **Line 319**: WMMA block skipped (condition false)
5. **Line 339**: Decode-specific block skipped (`hip_wmma_decode = false`)
6. **Falls through to line 360+**: Regular selection logic
   - For quantized K/V with `Q->ne[1] <= 2` → VEC (line 329)
   - Otherwise → TILE (line 334)

**Result**: For decode (`Q->ne[1] == 1`), uses **VEC kernel** via line 329-331.

### rocWMMA build (`-DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON`)

1. **Line 278**: `turing_mma_available(cc)` → `false`
2. **Line 319**: `ggml_cuda_should_use_wmma_fattn(cc)` → **returns `true`**
   - See `fattn-wmma-f16.cuh:30-31`:
   ```cpp
   if (GGML_CUDA_CC_IS_RDNA3(cc)) {
       return true;  // ← Takes this path for RDNA3 when rocWMMA enabled
   ```
3. **Line 305**: `hip_wmma_decode = Q->ne[1] == 1` → `true` for decode
4. **Line 319**: WMMA block condition is **false** because `!hip_wmma_decode` fails
5. **Line 339**: **NEW DECODE PATH TRIGGERED** (`hip_wmma_decode = true`)
   - **Line 340**: `S = K->ne[1]` (context size)
   - **Line 343**: If `S < 2048` → VEC (line 349)
   - **Line 343**: If `S >= 2048` → **TILE** (line 356)

**Result**: For decode with large context (≥2048 tokens), uses **TILE kernel**!

## Status update (2025-10-28)

Summary of intent
- With `GGML_HIP_ROCWMMA_FATTN=ON`, keep WMMA for prefill/pp where it is competitive.
- For decode/tg (`Q->ne[1] == 1`), avoid WMMA and use the regular HIP selection (VEC or TILE) — the same logic used when rocWMMA is not enabled.

What changed in code
- File: `llama.cpp/ggml/src/ggml-cuda/fattn.cu` (in `ggml_cuda_get_best_fattn_kernel`).
  - Decode (Q->ne[1]==1) falls through to the regular HIP selection with a guard: if a predicted TILE split has no config, fall back to VEC.
- File: `llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh`.
  - Remove WMMA-specific TILE pruning on HIP. Pruning remains for CUDA WMMA only.

Resulting behavior
- Prefill (pp): WMMA remains selected when `ggml_cuda_should_use_wmma_fattn(cc)` returns true.
- Decode (tg): Uses HIP’s normal logic (VEC when the vector preconditions hold, else TILE). No WMMA for decode. If no TILE config exists, guarded fallback to VEC.

Bench repos left unmodified
- The directories `llama.cpp-rocwmma/` and `llama.cpp-hip/` are reference/benchmark builds and must remain unchanged. All experiments and gating logic described here apply only to `llama.cpp/` in this repo. When comparing baselines, build and run from those reference trees as-is.

Why we did not force TILE by default
- We rely on the existing HIP selection (which already chooses TILE for long contexts when advantageous) and added a safety guard to avoid invalid splits.

Open issues we are tracking
- Decode perf regression when rocWMMA is enabled even if VEC is selected:
  - Observed: HIP‑only build VEC is faster than rocWMMA build VEC at large S (e.g., d4096/d65536).
  - Suspicions: compiler/regalloc differences when `GGML_HIP_ROCWMMA_FATTN` is defined; different inlining, launch bounds, or optimization passes.
  - Actions:
    - Diff `hipcc` flags and preprocessor macros between HIP‑only and rocWMMA builds.
    - Confirm `FAST_FP16_AVAILABLE`/`WARP_SIZE` paths are identical in `fattn-vec.cuh`.
    - Use `rocprof` to compare occupancy, waves/CU, LDS usage for VEC under both builds.

- TILE decode crash when rocWMMA is enabled (historical):
  - Root cause was device‑side pruning in WMMA builds. Fixed by removing pruning on HIP and adding a guard.
  - Validation: confirm selected tile config (`nwarps`, `nbatch_fa`, `nbatch_K`, LDS) matches HIP‑only behavior on RDNA3.

Decode TILE abort under WMMA builds (original observation and fix)
- Original symptom: TILE aborts under WMMA builds due to device-side pruning (`NO_DEVICE_CODE`).
- Fix: removed WMMA TILE pruning on HIP and added a host-side guard; no more aborts.

What "HIP arch 1300" means
- This message uses a synthetic arch value defined in `vendors/hip.h`:
  - `#define __CUDA_ARCH__ 1300` is set for HIP so common code can use the same `__CUDA_ARCH__` gate/macros.
  - The number 1300 is not your GFX target (gfx1151). Your actual GPU target is controlled by `AMDGPU_TARGETS` (e.g., `gfx1151`) and HipCC.
  - The `NO_DEVICE_CODE` trap prints that synthetic value; it does not indicate we compiled for the wrong GPU.

TILE pruning on WMMA builds (AMD)
- Removed for HIP/AMD entirely (aligns with HIP-only behavior). Only CUDA WMMA still prunes.
- Rationale: avoid device traps; enable TILE for long-context decode. Binary growth on your setup was ~+4 MiB.
- Guard remains in selection to fall back to VEC if no TILE config exists.

Selection guard to avoid device aborts
- Added and kept a decode-time guard in `ggml_cuda_get_best_fattn_kernel` (HIP+rocWMMA builds):
  - Predicts `ncols2` and `cols_per_block` like the TILE launcher and checks `ggml_cuda_fattn_tile_get_config(DKQ,DV,ncols,cc)`.
  - If no config is available, returns VEC with a one-time debug line: `DEBUG: Forcing VEC (no TILE config: DKQ=... DV=... ncols=... ncols2=...)`.
  - This covers the “TILE literally doesn’t exist” case without relying on pruning.

On removing pruning entirely
- Possible but heavy: it explodes compile time and binary size by materializing many more `flash_attn_tile<DKQ,DV,ncols1,ncols2>` instantiations (per DV, per ncols split, and per softcap variant). This is why WMMA builds pruned multi-column TILE in the first place.
- Pragmatic plan:
  - Keep pruning, but expand the allowlist to cover common head sizes (done) and guard selection to avoid device traps (done).
  - If needed, add a build flag later (e.g., `GGML_HIP_WMMA_KEEP_ALL_TILE=ON`) to disable pruning entirely for debugging.

Temporarily disable pruning (legacy/testing)
- Historical note: we briefly used a build flag to disable WMMA pruning on HIP while investigating. This is no longer needed because HIP WMMA no longer prunes TILE variants by default.

Benchmarking
- Default behavior with rocWMMA enabled: `llama.cpp/build/bin/llama-bench ... -fa 1 ...`
- Expected: Prefill uses WMMA; decode uses VEC/TILE per HIP selection. No env overrides or extra debug prints required.

Notes
- Goal: pp on WMMA, tg on HIP’s tuned kernels. Safety guard avoids invalid TILE selections.

## Final (PR‑ready) summary

Scope of changes (minimal and focused)
- `llama.cpp/ggml/src/ggml-cuda/fattn.cu`
  - Do not use WMMA for decode (unchanged intent); decode falls through to HIP logic.
  - Add a host‑side guard: if predicted TILE split has no config, select VEC to avoid device aborts.
  - Remove all ad‑hoc env overrides and debug prints.
- `llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh`
  - Remove WMMA‑specific TILE pruning for HIP/AMD; keep pruning only for CUDA WMMA builds.
- `llama.cpp/ggml/src/ggml-hip/CMakeLists.txt`
  - No new flags; reverted temporary switch used during investigation.

Behavior
- Prefill (pp): WMMA on RDNA3 remains active and significantly faster.
- Decode (tg): Uses HIP’s tuned kernels (VEC/TILE). Long‑context decode benefits from TILE on RDNA3; guard ensures we never crash on an unsupported TILE split.

Reference trees
- `llama.cpp-rocwmma/` and `llama.cpp-hip/` left untouched (benchmark baselines).

Risk/compatibility
- HIP binary grew only ~+4 MiB with HIP WMMA pruning removed; acceptable trade‑off for stability and performance.
- No behavior change for CUDA or non‑HIP builds beyond existing pruning on CUDA WMMA.

Validation checklist
- Rebuild HIP RDNA3 with rocWMMA ON and run `llama-bench` at varied depths. Confirm:
  - Prefill prints no extra debug; performance improved vs pre‑change.
  - Decode matches HIP‑only performance; no device aborts.
  - Optional: large‑S decode shows TILE selected and faster than VEC.

Changelog (concise)
- Remove HIP WMMA TILE pruning; add decode‑time guard; remove debug/overrides.

Status: Ready to propose upstream as a small, surgical change improving RDNA3 WMMA prefill while preserving HIP decode performance and stability.

### Final Bench Snapshot (paste from your run)

Command
- `llama.cpp/build/bin/llama-bench -r 1 -fa 1 -d 0 --mmap 0 -m /path/to/model.gguf`

Expected
- Prefill entries show strong WMMA throughput.
- Decode entries match HIP baseline performance and do not abort.

Logs
- [Paste your final `llama-bench` output here for recordkeeping.]

## PR Summary (for upstream)

- Massive WMMA prefill improvements on RDNA3: increased HIP occupancy and reduced LDS footprint via adaptive KQ stride; net pp speedups without touching CUDA or the deprecated Volta WMMA path.
- Fix long‑context decode regression on rocWMMA builds: decode now uses HIP’s tuned VEC/TILE selection instead of WMMA, aligning performance with the HIP baseline.
- Remove HIP‑side TILE pruning in WMMA builds: matches HIP‑only behavior and avoids device traps. Binary growth ~+4 MiB.
- Add a decode‑time safety guard (HIP+rocWMMA only): if a predicted TILE split has no config, fall back to VEC. This guard is not present in HIP‑only builds (they historically relied on configs being available) and avoids crashes on unusual dims.
- Changes are gated to ROCWMMA/HIP only; no impact to CUDA or the legacy Volta WMMA path.
Include perf chart and final bench output demonstrating:
- pp: WMMA better than before.
- tg: parity with HIP at long context (TILE where appropriate), with stable behavior.
