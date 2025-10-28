# llama.cpp CUDA vs HIP backends: architecture, precision, bottlenecks, and throughput estimates

This doc summarizes how llama.cpp runs inference on NVIDIA (CUDA) and AMD (HIP) GPUs, how kernels and precisions differ by architecture, where bottlenecks usually land, and how to estimate performance from hardware characteristics. It is grounded in the ggml backends and current code in this repo.

## Key takeaways

- HIP reuses the CUDA kernel sources via hipify and builds against rocBLAS/hipBLAS; most logic is shared, diverging based on device feature macros.
- Weight GEMMs dominate per‑token work and are typically memory‑bandwidth bound at inference (batch=1). Quantized INT8 weight GEMMs usually determine tg for short and medium context lengths; attention matmuls (FP16 path) take over at large contexts.
- NVIDIA Ampere/Ada: INT8 on Tensor Cores is very fast and usually hidden behind GDDR/HBM bandwidth; FP16 FlashAttention is fast. NVIDIA Turing: INT8 Tensor Cores are good but less capable than Ampere. NVIDIA Pascal/Volta: no INT8 Tensor Cores (dp4a/WMMA paths).
- AMD RDNA3: FP16 WMMA for attention is available (rocWMMA). INT8 performance is not dramatically higher than FP16 on RDNA3 (specs are same/CU/clock); tokens/s still benefits from INT8 mainly by halving bytes/weight (memory bound). AMD CDNA2/3 (MI210/MI300): MFMA INT8 is available; attention WMMA availability depends on rocWMMA version.
- Estimating tg: weight GEMMs are bandwidth bound, so tg ≈ memory_bandwidth / bytes_per_token_of_weights, until FP16 attention compute dominates at long contexts.


## How HIP is built from the CUDA backend

- HIP backend compiles the CUDA sources with HIP and links rocBLAS/hipBLAS. See: `llama.cpp/ggml/src/ggml-hip/CMakeLists.txt: file(GLOB GGML_SOURCES_ROCM "../ggml-cuda/*.cu")` and redefinitions in `llama.cpp/ggml/src/ggml-cuda/vendors/hip.h` mapping `cublas*` to `hipblas*`.
  - File refs: llama.cpp/ggml/src/ggml-hip/CMakeLists.txt, llama.cpp/ggml/src/ggml-cuda/vendors/hip.h
- Device‑feature selection is macro‑driven in `common.cuh`, enabling paths per architecture family (NVIDIA CCs; AMD GCN/CDNA/RDNA). HIP flips GGML_USE_HIP and AMD feature macros.
  - File ref: llama.cpp/ggml/src/ggml-cuda/common.cuh


## Kernel taxonomy and precisions in ggml‑cuda (+ HIP)

- **Quantized weight GEMMs**: `mmq.*` (mul_mat_q) and vector dot kernels in `vecdotq.cuh`.
  - Fused dequant + int dot product (dp4a or MMA/MFMA) with float accumulation and output in F32 (then cast/use F16 activation downstream).
  - DP4A path: NVIDIA ≥ sm_61, AMD RDNA2/3, earlier GCN via VEGA20; Tensor/MFMA path on NVIDIA ≥ Turing and AMD CDNA via MFMA.
  - File refs: llama.cpp/ggml/src/ggml-cuda/mmq.cu, llama.cpp/ggml/src/ggml-cuda/mmq.cuh, llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh
- **FP16 GEMMs and FlashAttention**: `fattn*` kernels with F16 matrix ops and F32 accumulations; backend selects MMA (Turing+/Ampere+), WMMA (Volta; RDNA3 via rocWMMA), or tile/vector fallback.
  - File refs: llama.cpp/ggml/src/ggml-cuda/fattn.cu, llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh, llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cuh
- **Elementwise and small ops**: rope, norm, softmax, etc., mostly F16 storage with F32 accumulations; generally memory‑traffic limited.
  - File refs: llama.cpp/ggml/src/ggml-cuda/rope.cu, llama.cpp/ggml/src/ggml-cuda/norm.cu, llama.cpp/ggml/src/ggml-cuda/softmax.cu


## Inference pipeline: ops, precisions, dominant kernels

For a LLaMA‑style block with model dim d, heads h, head_dim dk=d/h, FFN dim df, sequence so far S, generating 1 new token:

| Operation | Precision | MACs | Dominant Kernel | Notes |
| --- | --- | --- | --- | --- |
| Embedding lookup | FP16 storage | — | Memory gather | Bandwidth bound |
| RMSNorm | F32 accum | — | Custom | Memory bound, negligible time |
| Q, K, V projections | INT8×FP16→INT32→F32 | 3×(d × d) | mmq (int8) or mmf (fp16) | Weight GEMMs; fallback to cuBLAS/hipBLAS |
| RoPE | F16, F32 intermediates | — | Custom | Elementwise, negligible |
| Attention QK | FP16→F32 accum | dk × S per head | FlashAttention (MMA/WMMA) or tile/vector | Cost scales with S |
| Attention softmax | F32 | — | Custom | Memory bound |
| Attention PV | FP16→F32 accum | S × dk per head | FlashAttention (MMA/WMMA) or tile/vector | Cost scales with S |
| Output projection | INT8×FP16→INT32→F32 | d × d | mmq (int8) | Weight GEMM |
| MLP gate/up | INT8×FP16→INT32→F32 | 2×(d × df) | mmq (int8) | Weight GEMMs |
| MLP activation | F16/F32 | — | Custom | Elementwise, memory bound |
| MLP down | INT8×FP16→INT32→F32 | df × d | mmq (int8) | Weight GEMM |
| RMSNorm | F32 accum | — | Custom | Memory bound, negligible |
| LM head | INT8/FP16 | vocab_size × d | mmq/mmf or BLAS | Usually cached on GPU, bandwidth‑bound |

**Dominant work**: Weight GEMMs (INT8 mmq) for QKV/O/MLP when S is small to moderate; attention FP16 work dominates as S grows.


## FLOPs and precision share (typical, weight‑only quantized LLaMA)

Let MACs = multiply‑accumulates; 1 MAC ≈ 2 FLOPs. Single‑token, per layer:

- Weight GEMMs MACs ≈ 4·d² (QKV + O) + 3·d·df (SwiGLU has 2 up/gate + 1 down)
- Attention MACs ≈ S·d (QK) + S·d (PV) = 2·S·d

Precision split (dominant math unit):

- Weight GEMMs: INT8 dot (dp4a/MMA/MFMA) with F32 accum; time dominated by memory traffic of int8 weights.
- Attention QK/PV: FP16 MMA/WMMA with F32 accum; compute‑bound at large S, else memory‑bound.
- Elementwise/norm/softmax: FP16 storage, F32 accum; small cost.

Approximate fraction of MACs by category (varies by model size d, df and S):

- Short context (S ≪ d, e.g., S ≤ 256 for 7B): weight GEMMs ≳ 90–95% of MACs; attention ≲ 5–10%.
- Mid context (S ~ d, e.g., S ≈ 4k head_dim spread across layers): weight GEMMs ~ 70–85%; attention ~ 15–30%.
- Long context (S ≫ d): attention can exceed 50% of MACs; eventually compute‑bound on FP16.

These are guidance bands; exact shares should be measured per export (quant type, kv cache type) and sequence schedule.


## Architecture tables: components, precision, methods, and bottlenecks

**Table notes**:
- Shares are indicative for LLaMA‑style, weight‑only INT8 export, d≈4096, df≈11008, L≈32.
- "Share (tg)" corresponds to S≈2k (decode); "Share (pp)" corresponds to S≈8k (prefill). Shares reflect MACs, not time; time shares vary by arch and kernel efficiency.
- Library "Custom" means llama.cpp ggml kernels (mmq/FA/tile/vec). WMMA refers to nvcuda::wmma (NVIDIA) or rocWMMA (AMD).

### NVIDIA Pascal (sm_60/61)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16 storage, F32 accum | elementwise | Custom | Small, memory bound |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | dp4a | Custom (mmq dp4a) | No INT8 Tensor Cores; sm_61 needed for dp4a |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | tile/vector GEMV/GEMM | Custom | No WMMA/MMA; FA uses non‑tensor path |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom | Memory bound |

**Implementation**:
- INT8 via dp4a on SM; no INT8 Tensor Cores (requires sm_61+ for dp4a; sm_60/GP100 lacks dp4a, sm_61/GP102-104-106 has it).
- FP16 arithmetic supported but no MMA; attention uses tile/vector kernels (slower).
- File refs: llama.cpp/ggml/src/ggml-cuda/common.cuh (feature detection), llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh (dp4a)

**Bottlenecks**:
- Weight GEMMs memory‑bound at all context lengths.
- Attention slower than Volta+ due to lack of MMA/WMMA.
- Consider FP16 or higher‑bit quant (Q8_0) if dp4a utilization is poor.

### NVIDIA Volta (sm_70)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16/F32 | elementwise | Custom |  |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | dp4a | Custom (mmq dp4a) | INT8 via dp4a |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | WMMA | WMMA (nvcuda) | Legacy WMMA layout kept for Volta |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom |  |

**Implementation**:
- WMMA FP16 available for attention; INT8 still via dp4a (no INT8 WMMA/MMA).
- FlashAttention WMMA fallback path maintained for Volta.
- File ref: llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cu

**Bottlenecks**:
- Weight GEMMs remain memory‑bound.
- Attention faster than Pascal but slower than Turing+ (no MMA).
- No INT8 Tensor Cores; dp4a still helps.

### NVIDIA Turing/Ampere/Ada/Hopper (sm_75/80/86/89/90)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16/F32 | elementwise | Custom |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | MMA int8 (mma.sync); cp.async on Ampere+ | Custom (mmq MMA) | Strong INT8 Tensor Cores; Hopper FP8 unused here |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | MMA FP16 | Custom FA (mma.sync) | High FA perf across these gens |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom |

**Implementation**:
- Turing (sm_75): Tensor Cores support MMA FP16 and INT8. `TURING_MMA_AVAILABLE` enables mma.sync paths.
- Ampere/Ada (sm_80/86/89): Adds TF32/FP16/INT8 MMA, cp.async, better FlashAttention. `AMPERE_MMA_AVAILABLE` and `CP_ASYNC_AVAILABLE` drive variants.
- Hopper (sm_90): FP8 not used by llama.cpp kernels today; otherwise similar to Ampere.
- File refs: llama.cpp/ggml/src/ggml-cuda/mma.cuh (MMA implementations), llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh (FlashAttention)

**Bottlenecks**:
- Weight GEMMs saturate HBM bandwidth; INT8 Tensor Cores very fast but memory‑bound.
- Attention compute only dominates at large S (long context).
- **Tuning**: Ensure FA enabled (default). Keep kv‑cache in FP16 unless memory constrained; int8 KV hurts attention quality and does not hit Tensor Cores. Expect tg scaling with HBM bandwidth for INT8 weight‑only quant.

### AMD VEGA20 (GCN)/RDNA2 (gfx906/gfx10.3)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16/F32 | elementwise | Custom | Wave64 on VEGA20; Wave32 on RDNA2 |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | dp4a | Custom (mmq dp4a) | No WMMA on either |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | tile/vector | Custom | Attention non‑tensor; slower kernels |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom |

**Implementation**:
- VEGA20 (gfx906): dp4a available; wave size 64. GCN architecture.
- RDNA2 (gfx103x): dp4a available; wave size 32. No WMMA/MFMA.
- No tensor/matrix acceleration for attention; uses tile/vector fallback.
- File ref: llama.cpp/ggml/src/ggml-cuda/common.cuh (wave size detection)

**Bottlenecks**:
- Weight GEMMs memory‑bound; INT8 reduces bytes but compute not accelerated beyond dp4a.
- Attention uses tile/vector and can be a bottleneck at mid/long S. Slower than RDNA3/NVIDIA with tensor cores.
- **Tuning**: INT8 still reduces memory bytes, which helps tg at small S. Consider upgrading to RDNA3+ for attention performance.

### AMD RDNA3/RDNA4 (gfx11.x/gfx12.x)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16/F32 | elementwise | Custom |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | dp4a | Custom (mmq dp4a) | INT8 compute ~ FP16; bytes dominate |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | WMMA FP16 | WMMA (rocWMMA) | Enable GGML_HIP_ROCWMMA_FATTN; RDNA4 requires rocWMMA ≥ 2.0 |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom |

**Implementation**:
- WMMA (FP16) available via rocWMMA for attention; significant improvement over RDNA2.
- INT8 mostly uses dp4a (no INT8 MFMA on RDNA). INT8 "TOPS" ≈ FP16 TFLOPs in practice.
- Wave size 32.
- File refs: llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cuh (WMMA detection and version checks)

**Bottlenecks**:
- Weight GEMMs remain memory‑bound; INT8 compute is not a big win vs FP16, but INT8 halves bytes → better tg when memory‑bound.
- Attention benefits from WMMA but still slower than NVIDIA Ampere+ MMA.
- **Tuning**: Use FA via rocWMMA for attention (enable GGML_HIP_ROCWMMA_FATTN). INT8 mostly saves memory bytes; choose quantization to fit model in VRAM and reduce bytes/weight.

### AMD CDNA1/2/3 (MI100/MI200/MI300)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16/F32 | elementwise | Custom |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | MFMA int8 | Custom (mmq MFMA) | Strong INT8 MFMA across CDNA |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | WMMA FP16 | WMMA (rocWMMA) | WMMA availability/version‑dependent; fallback to vector/tile when needed |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom |

**Implementation**:
- MFMA available and fast, including INT8 MMA acceleration.
- Attention WMMA depends on rocWMMA version: rocWMMA v2.0.0 specifically is broken on CDNA2/3; versions before or after work correctly.
- Wave size 64 (GFX9).
- File refs: llama.cpp/ggml/src/ggml-cuda/mma.cuh (MFMA int8 implementation), llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cuh (version checking)

**Bottlenecks**:
- At small S, weight memory dominates despite fast MFMA.
- At larger S, attention compute can dominate but MFMA helps keep it efficient.
- **Tuning**: Use MFMA INT8 mmq. Confirm rocWMMA version for FA; if buggy (v2.0.0), use vector/tile attention or upgrade/downgrade rocWMMA. Both INT8 (MFMA) and FP16 are strong; still weight path bounded by memory for small S.

### Unquantized variants

- If weights are FP16/BF16, Projections/MLP rows become FP16 GEMM; library can be Custom (mmf kernels) or cuBLAS/hipBLAS depending on shape/flags.
- Attention precision/method remains as above; FP16 path dominates at large S for pp.


## tg (decode) vs pp (prefill)

**Definitions**:
- **tg (decode)**: single‑step token generation at batch≈1. Arithmetic intensity is low; performance is usually limited by weight fetch bandwidth. Attention cost scales with S but remains smaller until S becomes large.
- **pp (prefill)**: processing a prompt/context with large effective batch/sequence. Arithmetic intensity is high; attention math and large FP16 GEMMs dominate and can be compute‑bound on capable Tensor/Matrix units.

**Implications**:
- On tg, optimizing bytes/weight (quantization, cache locality, stream‑k thresholds) yields the biggest gains.
- On pp, optimizing MMA/WMMA kernels and ensuring high sustained FP16 throughput (and FA enabled) is critical; INT8 helps less because weight fetch is amortized across many tokens.


## Backend decision points (kernel selection)

**Quantized weight GEMMs (mmq)**:
- Chooses custom mmq vs BLAS based on arch and shape. On NVIDIA with Turing+ or dp4a, mmq is used widely. On AMD CDNA3 it prefers mmq; otherwise falls back to hipBLAS for some shapes.
- File ref: llama.cpp/ggml/src/ggml-cuda/mmq.cu (ggml_cuda_should_use_mmq function)

**Attention kernel selection**:
- Chooses among tile/vector, WMMA‑F16, or MMA‑F16 based on head size, batch, mask/GQA, and architecture.
- File ref: llama.cpp/ggml/src/ggml-cuda/fattn.cu (ggml_cuda_get_best_fattn_kernel function)


## Throughput (tg) estimation: a practical roofline

Goal: estimate tokens/s for batch=1, given hardware bandwidth/throughput and model export.

**1) Weight‑GEMM bandwidth limiter (dominant at small to mid S)**:
- bytes_per_token_weights ≈ bytes_per_param_used_per_token. For LLaMA‑style blocks, nearly all linear weights are touched once per token:
  - Per layer, parameters contributing per token ≈ (4·d² + 3·d·df) weights. Multiply by bytes/weight (1 for INT8, 2 for FP16/BF16, 4 for FP32).
  - Add embedding and LM head if on GPU and not cached; usually small vs layers.
- Device effective bandwidth BW_eff (HBM bandwidth × utilization). For modern GPUs, 50–80% of peak is typical for these access patterns.
- Estimate: tg_weight ≈ BW_eff / (L · bytes_per_layer_per_token)

**2) Attention compute limiter (dominant at large S)**:
- MACs_attn_per_token_per_layer ≈ 2·S·d (QK + PV); compute on FP16 MMA/WMMA with F32 accum.
- Let T_FP16 be sustained FP16 Tensor (or WMMA/MFMA) throughput. Estimate time per layer for attention: t_attn ≈ (2·S·d · 2 FLOPs/MAC) / T_FP16.

**3) Combine**:
- Time per token ≈ L·(t_weight + t_attn + t_misc). t_misc is small (norm/rope/softmax), often < 5–10% unless FA is disabled.
- tg ≈ 1 / time_per_token. In the common regime, time_weight dominates until S grows enough that t_attn is comparable.

**Rules of thumb**:
- **NVIDIA Ampere/Ada**: Assume weight path fully memory‑bound and INT8 halves bytes vs FP16, so INT8 ≈ 2× FP16 tg when the weight path dominates. Attention benefits strongly from FA + MMA.
- **AMD RDNA3**: INT8 vs FP16 compute speed similar; tg uplift from INT8 comes from halving weight bytes. If attention dominates (very large S), INT8 advantage shrinks.
- **AMD CDNA2/3**: Both INT8 (MFMA) and FP16 are strong; still weight path bounded by memory for small S.


## Profiling and validation

**llama.cpp perf**:
- Build with CUDA/HIP and run with perf flags to get per‑op timings; compare with toggles like `--no-flash-attn`, `--parallel 1`, kv cache type, and quant types to see shifts.

**NVIDIA profiling**:
- Use Nsight Systems/Compute to attribute kernels and memory vs compute utilization; verify dp4a vs MMA utilization and DRAM BW.

**AMD profiling**:
- Use rocprof/Omniperf; inspect MFMA utilization (CDNA) and DRAM BW; check wave occupancy for RDNA.


## Worked example: 7B‑class INT8, short context

Model: d=4096, df≈11008, L=32, S small

**Calculation**:
- Per layer weights used ≈ 4·d² + 3·d·df ≈ 4·(16.8M) + 3·(45.1M) ≈ 200M weights
- At INT8: ~200 MB per layer per 1000 tokens
- Bytes per token across 32 layers ≈ 200M bytes/layer ÷ 1000 tokens × 32 ≈ ~6.4 MB per token (rough; excludes caching/reuse)
- If effective HBM BW ≈ 800 GB/s (example), tg_weight ≈ 800e9 / 6.4e6 ≈ 125k tokens/s theoretical upper bound for weight path alone

**Reality**: Real tg is far lower due to parallelism/latency/launch overhead, and because attention + misc ops add time. Use perf to calibrate BW_eff and per‑op weights actually touched. This sketch intentionally overestimates to be conservative on identifying the bottleneck: bandwidth dominates.
In practice, weight tile reuse, caching, and kernel overlap further reduce effective bytes per token compared to this upper bound.


## Build flags and knobs

**CUDA/HIP common**:
- `GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS`: force custom kernels vs BLAS
- `GGML_CUDA_NO_FA`: disable FlashAttention (for baseline comparisons)
- File refs: llama.cpp/ggml/src/ggml-cuda/mmq.cu, llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt

**HIP specific**:
- `GGML_HIP_ROCWMMA_FATTN`: enable WMMA via rocWMMA for attention
- `GGML_HIP_NO_MMQ_MFMA`: disable MFMA path
- `GGML_HIP_GRAPHS`: enable HIP graphs
- `GGML_HIP_NO_VMM` (HIP) and `GGML_CUDA_NO_VMM` (CUDA): disable VMM; HIP VMM is disabled by default via CMake option, enable by passing `-DGGML_HIP_NO_VMM=OFF`
- File ref: llama.cpp/ggml/src/ggml-hip/CMakeLists.txt


## Future optimization opportunities

- **Weight tile reuse**: Improve memory residency/reuse of weight tiles in mmq for batch>1 and grouped rows; batch=1 inference is bandwidth‑bound.
- **RDNA attention**: Invest in vector/tile FA kernels and shared‑mem tiling to close the gap to tensor‑accelerated attention on NVIDIA/CDNA.
- **KV cache format**: FP16 vs Q8_0 tradeoffs (memory vs attention FP16 compute). Long‑context runs can become attention‑bound; choose KV precision accordingly.
- **Stream‑K tuning**: Already used on NVIDIA Volta+ and CDNA; improve decomposition thresholds for better occupancy on tall/wide GEMMs.


## Measuring real‑world performance

### Baseline tokens/s with llama-bench
- Run `llama-bench` on the standard llama.cpp LLaMA-2 7B quant configs (INT8 and FP16 baselines) for both `decode` and `prefill` passes. Capture the reported tokens/s and the optional per-op timings (build with `-DLLAMA_CUDA_DEBUG=ON` to emit kernel timings).
- Derive an empirical effective bandwidth per token: `bytes_touched_per_layer / measured_time`. Compare this to the roofline estimate to quantify how much reuse, caching, and overlap shrink the effective bytes vs the theoretical 6.4 MB/token.
- When comparing architectures, normalize by theoretical HBM/throughput to express results as % of peak compute or % of peak bandwidth; keep precisions separate (INT8 vs FP16) to avoid mixing tensor-unit ceilings.

### Kernel microbenchmarks and GEMM shape sensitivity
- `tests/test-backend-ops perf` exercises individual kernels (mmq, fattn, mmf) with tunable shapes and precisions. Use it to sweep matrix aspect ratios that match production (tall/skinny, wide/tall, small batch) and record sustained GFLOPs/GB/s.
- Not all GEMMs are equal: llama.cpp matmuls span from square projections to skinny attention blocks. Measure how throughput drops outside TensorCore-friendly tile sizes to understand which paths dominate time even when aggregate MAC counts are similar.
- Cross-check microbench results with the shapes invoked during `llama-bench` by logging `ne` dimensions; this highlights when e2e stalls stem from shape-specific underutilization vs scheduler gaps.

### Hardware profiling for ground truth
- **NVIDIA**: Nsight Compute/Nsight Systems provide per-kernel `dram__bytes`, TensorCore utilization, occupancy, and timeline gaps. Export CSV traces and align with llama-bench timers to attribute time to memory stalls vs compute.
- **AMD**: rocprof / Omniperf report MFMA occupancy, global memory bytes, and wave occupancy. Enable rocWMMA counters where available to confirm attention WMMA usage on RDNA3/CDNA.
- From profiler counters compute effective bandwidth and math throughput per kernel, then aggregate per token/layer. This validates the empirical bytes/token numbers and shows where overlap hides latency.

### Interpreting results and known limitations
- Warmup effects and cache residency skew early tokens; discard the first N iterations or run long prompts before sampling metrics.
- Mixed precision complicates “GFLOPs” ceilings—track INT8 vs FP16 ops separately and compare to the correct hardware peak for each.
- HIP profiling can miss counters on some SKUs; fall back to timeline traces when utilization numbers are unavailable.
- Dynamic kernel selection (mmq vs BLAS, WMMA vs tile) depends on shape and build flags; log the chosen kernels during bench runs so measurements map to code paths.
- Even with detailed data, overlap between compute and memory transfers means the simple roofline is an upper bound. The methodologies above turn the gap between theory and practice into measurable numbers you can track across code revisions.
