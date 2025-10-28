# llama.cpp CUDA vs HIP backends: architecture, precision, bottlenecks, and throughput estimates

This doc summarizes how llama.cpp runs inference on NVIDIA (CUDA) and AMD (HIP) GPUs, how kernels and precisions differ by architecture, where bottlenecks usually land, and how to estimate token throughput (tg) from hardware characteristics. It is grounded in the ggml backends and current code in this repo.

Key takeaways up front

- HIP reuses the CUDA kernel sources via hipify and builds against rocBLAS/hipBLAS; most logic is shared, diverging based on device feature macros.
- Weight GEMMs dominate per‑token work and are typically memory‑bandwidth bound at inference (batch=1). Quantized INT8 weight GEMMs usually determine tg for short and medium context lengths; attention matmuls (FP16 path) take over at large contexts.
- NVIDIA Ampere/Ada: INT8 on Tensor Cores is very fast and usually hidden behind HBM bandwidth; FP16 FlashAttention is fast. NVIDIA Turing: INT8 Tensor Cores are good but less capable than Ampere. NVIDIA Pascal/Volta: no INT8 Tensor Cores (dp4a/WMMA paths).
- AMD RDNA3: FP16 WMMA for attention is available (rocWMMA). INT8 performance is not dramatically higher than FP16 on RDNA3; tokens/s still benefits from INT8 mainly by halving bytes/weight (memory bound). AMD CDNA2/3 (MI210/MI300): MFMA INT8 is available; attention WMMA availability depends on rocWMMA version.
- Estimating tg: weight GEMMs are bandwidth bound, so tg ≈ memory_bandwidth / bytes_per_token_of_weights, until FP16 attention compute dominates at long contexts.


## How HIP is built from the CUDA backend

- HIP backend compiles the CUDA sources with HIP and links rocBLAS/hipBLAS. See: `llama.cpp/ggml/src/ggml-hip/CMakeLists.txt: file(GLOB GGML_SOURCES_ROCM "../ggml-cuda/*.cu")` and redefinitions in `llama.cpp/ggml/src/ggml-cuda/vendors/hip.h` mapping `cublas*` to `hipblas*`.
  - File refs: llama.cpp/ggml/src/ggml-hip/CMakeLists.txt:41, llama.cpp/ggml/src/ggml-hip/CMakeLists.txt:53, llama.cpp/ggml/src/ggml-cuda/vendors/hip.h:33
- Device‑feature selection is macro‑driven in `common.cuh`, enabling paths per architecture family (NVIDIA CCs; AMD GCN/CDNA/RDNA). HIP flips GGML_USE_HIP and AMD feature macros:
  - File refs: llama.cpp/ggml/src/ggml-cuda/common.cuh:60, llama.cpp/ggml/src/ggml-cuda/common.cuh:246


## Kernel taxonomy and precisions in ggml‑cuda (+ HIP)

- Quantized weight GEMMs: `mmq.*` (mul_mat_q) and vector dot kernels in `vecdotq.cuh`.
  - Fused dequant + int dot product (dp4a or MMA/MFMA) with float accumulation and output in F32 (then cast/use F16 activation downstream).
  - DP4A path: NVIDIA ≥ sm_61, AMD RDNA2/3, earlier GCN via VEGA20; Tensor/MFMA path on NVIDIA ≥ Turing and AMD CDNA via MFMA.
  - File refs: llama.cpp/ggml/src/ggml-cuda/mmq.cu:1, llama.cpp/ggml/src/ggml-cuda/mmq.cuh:1, llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:1
- FP16 GEMMs and FlashAttention: `fattn*` kernels with F16 matrix ops and F32 accumulations; backend selects MMA (Turing+/Ampere+), WMMA (Volta; RDNA3 via rocWMMA), or tile/vector fallback.
  - File refs: llama.cpp/ggml/src/ggml-cuda/fattn.cu:1, llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:1, llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cuh:1
- Elementwise and small ops: rope, norm, softmax, etc., mostly F16 storage with F32 accumulations; generally memory‑traffic limited.
  - File refs: llama.cpp/ggml/src/ggml-cuda/rope.cu:1, llama.cpp/ggml/src/ggml-cuda/norm.cu:1, llama.cpp/ggml/src/ggml-cuda/softmax.cu:1


## Which architectures get which math units

The selection logic is in `common.cuh` and `mmq.cuh` feature checks.

- NVIDIA
  - Pascal (sm_60/61): FP16 arithmetic supported; INT8 via dp4a; no INT8 Tensor Cores. Attention uses tile/vector or WMMA on Volta only.
  - Volta (sm_70): WMMA FP16 exists; INT8 still via dp4a; FlashAttention WMMA fallback path is kept for Volta.
    - File ref: llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cu:1
  - Turing (sm_75): Tensor Cores support MMA FP16 and INT8. `TURING_MMA_AVAILABLE` enables mma.sync paths for int8/f16.
    - File ref: llama.cpp/ggml/src/ggml-cuda/mma.cuh:146
  - Ampere/Ada (sm_80/86/89): Adds TF32/FP16/INT8 MMA, cp.async, better FlashAttention. `AMPERE_MMA_AVAILABLE` and `CP_ASYNC_AVAILABLE` drive variants.
  - Hopper (sm_90): FP8 not used by llama.cpp kernels today; otherwise similar to Ampere for our purposes.
- AMD
  - VEGA20 (GCN, gfx906): dp4a available; wave size 64. No WMMA; attention uses tile/vector.
  - RDNA2 (gfx103x): dp4a available; no WMMA; attention tile/vector.
  - RDNA3 (gfx11xx): WMMA (FP16) available via rocWMMA for attention; INT8 mostly uses dp4a (no int8 MFMA on RDNA). Wave size 32.
    - File ref: llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cuh:15
  - CDNA1/2/3 (MI100/MI200/MI300): MFMA available, including INT8 MMA; attention WMMA depends on rocWMMA version (some versions broken on CDNA2/3).
    - File refs: llama.cpp/ggml/src/ggml-cuda/mma.cuh:500, llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cuh:26


## Backend decision points (what runs where)

- Quantized weight GEMMs (mmq)
  - Chooses custom mmq vs BLAS based on arch and shape. On NVIDIA with Turing+ or dp4a, mmq is used widely. On AMD CDNA3 it prefers mmq; otherwise falls back to hipBLAS for some shapes.
  - File ref: llama.cpp/ggml/src/ggml-cuda/mmq.cu:236
- Attention kernel selection
  - Chooses among tile/vector, WMMA‑F16, or MMA‑F16 based on head size, batch, mask/GQA, and architecture.
  - File refs: llama.cpp/ggml/src/ggml-cuda/fattn.cu:220, llama.cpp/ggml/src/ggml-cuda/fattn.cu:288


## Inference pipeline: ops, precisions, dominant kernels

For a LLaMA‑style block with model dim d, heads h, head_dim dk=d/h, FFN dim df, sequence so far S, 1 new token:

- Input embedding lookup
  - Precision: typically FP16 storage (varies by model export). Memory gather; bandwidth bound.
- RMSNorm
  - F32 accumulation; memory bound; tiny share of time.
- Q, K, V projections
  - Weight GEMMs, 3×(d × d) MACs; precision: INT8×FP16→INT32 accumulate→F32 output (then F16 activations), or FP16 if unquantized.
  - Dominant kernel: mmq (int8), mmf (fp16), optional cuBLAS/hipBLAS fallback.
- RoPE
  - Elementwise F16 with F32 intermediates; negligible.
- Attention
  - QK: FP16 matmul (dk × S per head), acc F32; kernel: FlashAttention (mma/wmma) or tile/vector.
  - Softmax + mask: F32 reduce/exp; memory bound.
  - PV: FP16 matmul (S × dk per head), acc F32; same kernel family as QK.
- Output projection (O)
  - Weight GEMM, (d × d) MACs; INT8 path (mmq) dominates for quantized weights.
- MLP (SwiGLU)
  - Gate/up: 2×(d × df) GEMMs (weights quantized ⇒ INT8 mmq).
  - Activation + elementwise gate: F16/F32; memory bound.
  - Down: (df × d) GEMM; INT8 mmq.
- Final RMSNorm, LM head
  - RMSNorm: small, memory bound. LM head may be INT8/FP16 depending on export; can be large for vocab size but typically cached on GPU and bandwidth‑bound.

Dominant work: weight GEMMs (INT8 mmq) for QKV/O/MLP when S is small to moderate; attention FP16 work dominates as S grows.


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


## Architecture tables: math units, kernel picks, and practical bottlenecks

NVIDIA

- Pascal (sm_60/61)
  - INT8: dp4a on SM; no int8 Tensor Cores.
  - FP16: fast FP16 arithmetic (no MMA), attention tile/vector kernels.
  - Bottlenecks: weight GEMMs memory‑bound; attention slower due to no MMA.
- Volta (sm_70)
  - INT8: dp4a; FP16 WMMA available; legacy WMMA path for attention.
  - Bottlenecks: still weight memory; attention faster than Pascal.
- Turing (sm_75)
  - INT8: mma.sync int8; FP16 MMA; attention: MMA.
  - Bottlenecks: weight memory dominates at small S; attention compute moderate.
- Ampere/Ada (sm_80/86/89)
  - INT8: strong Tensor Core throughput; cp.async; FA MMA kernel widely used.
  - Bottlenecks: weight GEMMs saturate HBM; attention compute only dominates at large S.
- Hopper (sm_90)
  - Similar to Ampere for INT8/FP16 paths used by llama.cpp; FP8 not used here.

AMD

- VEGA20 (gfx906)
  - INT8: dp4a; wave64. No WMMA; attention tile/vector.
  - Bottlenecks: weight memory; attention slower kernels.
- RDNA2 (gfx10.3)
  - INT8: dp4a; wave32; no WMMA; attention tile/vector.
  - Bottlenecks: weight memory; attention slower than RDNA3.
- RDNA3 (gfx11.x)
  - INT8: dp4a; INT8 “TOPS” ≈ FP16 TFLOPs in practice; WMMA FP16 available for attention via rocWMMA.
  - Bottlenecks: weight GEMMs remain memory‑bound; INT8 compute is not a big win vs FP16, but INT8 halves bytes → better tg when memory‑bound.
- CDNA1/2/3 (MI100/MI200/MI300)
  - INT8: MFMA available and fast; attention WMMA via rocWMMA subject to version quirks on CDNA2/3.
  - Bottlenecks: at small S, weight memory; at larger S, attention compute can dominate but MFMA helps.


## tg (decode) vs pp (prefill)

Definitions

- tg (decode): single‑step token generation at batch≈1. Arithmetic intensity is low; performance is usually limited by weight fetch bandwidth. Attention cost scales with S but remains smaller until S becomes large.
- pp (prefill): processing a prompt/context with large effective batch/sequence. Arithmetic intensity is high; attention math and large FP16 GEMMs dominate and can be compute‑bound on capable Tensor/Matrix units.

Implications

- On tg, optimizing bytes/weight (quantization, cache locality, stream‑k thresholds) yields the biggest gains.
- On pp, optimizing MMA/WMMA kernels and ensuring high sustained FP16 throughput (and FA enabled) is critical; INT8 helps less because weight fetch is amortized across many tokens.


## Throughput (tg) estimation: a practical roofline

Goal: estimate tokens/s for batch=1, given hardware bandwidth/throughput and model export.

1) Weight‑GEMM bandwidth limiter (dominant at small to mid S)

- bytes_per_token_weights ≈ bytes_per_param_used_per_token. For LLaMA‑style blocks, nearly all linear weights are touched once per token:
  - Per layer, parameters contributing per token ≈ (4·d² + 3·d·df) weights. Multiply by bytes/weight (1 for INT8, 2 for FP16/BF16, 4 for FP32).
  - Add embedding and LM head if on GPU and not cached; usually small vs layers.
- Device effective bandwidth BW_eff (HBM bandwidth × utilization). For modern GPUs, 50–80% of peak is typical for these access patterns.
- Estimate: tg_weight ≈ BW_eff / (L · bytes_per_layer_per_token)

2) Attention compute limiter (dominant at large S)

- MACs_attn_per_token_per_layer ≈ 2·S·d (QK + PV); compute on FP16 MMA/WMMA with F32 accum.
- Let T_FP16 be sustained FP16 Tensor (or WMMA/MFMA) throughput. Estimate time per layer for attention: t_attn ≈ (2·S·d · 2 FLOPs/MAC) / T_FP16.

3) Combine

- Time per token ≈ L·(t_weight + t_attn + t_misc). t_misc is small (norm/rope/softmax), often < 5–10% unless FA is disabled.
- tg ≈ 1 / time_per_token. In the common regime, time_weight dominates until S grows enough that t_attn is comparable.

Rules of thumb

- NVIDIA Ampere/Ada: Assume weight path fully memory‑bound and INT8 halves bytes vs FP16, so INT8 ≈ 2× FP16 tg when the weight path dominates. Attention benefits strongly from FA + MMA.
- AMD RDNA3: INT8 vs FP16 compute speed similar; tg uplift from INT8 comes from halving weight bytes. If attention dominates (very large S), INT8 advantage shrinks.
- AMD CDNA2/3: Both INT8 (MFMA) and FP16 are strong; still weight path bounded by memory for small S.


## Where the code enforces these choices (selected references)

- Architecture features and availability checks
  - File ref: llama.cpp/ggml/src/ggml-cuda/common.cuh:240
- INT8 MMQ kernel tiling/paths for NVIDIA vs AMD (dp4a vs MMA/MFMA)
  - File refs: llama.cpp/ggml/src/ggml-cuda/mmq.cuh:1, llama.cpp/ggml/src/ggml-cuda/mmq.cu:236
- AMD MFMA INT8
  - File ref: llama.cpp/ggml/src/ggml-cuda/mma.cuh:500
- FlashAttention kernel selection
  - File refs: llama.cpp/ggml/src/ggml-cuda/fattn.cu:220, llama.cpp/ggml/src/ggml-cuda/fattn.cu:304
- WMMA availability for HIP (RDNA3, CDNA via rocWMMA gating)
  - File ref: llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cuh:26


## Practical bottlenecks and suggestions by architecture

- NVIDIA Ampere/Ada
  - Ensure FA enabled (default). Keep kv‑cache in FP16 unless memory constrained; int8 KV hurts attention quality and does not hit Tensor Cores.
  - Expect tg scaling with HBM bandwidth for INT8 weight‑only quant.
- NVIDIA Turing
  - Similar, but some kernels less optimized; still strong INT8 path.
- NVIDIA Pascal/Volta
  - No INT8 Tensor Cores; dp4a still helps. Consider FP16 or higher‑bit quant (Q8_0) if dp4a utilization is poor.
- AMD RDNA2
  - No WMMA; attention uses tile/vector and can be a bottleneck at mid/long S. INT8 still reduces memory bytes, which helps tg when small S.
- AMD RDNA3
  - Use FA via rocWMMA for attention. INT8 mostly saves memory bytes; choose quantization to fit model in VRAM and reduce bytes/weight.
- AMD CDNA2/3
  - Use MFMA INT8 mmq. Confirm rocWMMA version for FA; if buggy, use vector/tile attention or specific build flags.


## Measuring the % breakdown and validating estimates

- llama.cpp perf
  - Build with CUDA/HIP and run with perf flags to get per‑op timings; compare with toggles like `--no-flash-attn`, `--parallel 1`, kv cache type, and quant types to see shifts.
- NVIDIA profiling
  - Use Nsight Systems/Compute to attribute kernels and memory vs compute utilization; verify dp4a vs MMA utilization and DRAM BW.
- AMD profiling
  - Use rocprof/Omniperf; inspect MFMA utilization (CDNA) and DRAM BW; check wave occupancy for RDNA.


## Worked example (7B‑class, weight‑only INT8, short context)

- d=4096, df≈11008, L=32, S small
- Per layer weights used ≈ 4·d² + 3·d·df ≈ 4·(16.8M) + 3·(45.1M) ≈ 200M weights; at INT8 → ~200 MB per layer per 1000 tokens.
- Bytes per token across 32 layers ≈ 200M bytes/layer ÷ 1000 tokens × 32 ≈ ~6.4 MB per token (very rough; excludes caching/reuse and small ops).
- If effective HBM BW ≈ 800 GB/s (example), tg_weight ≈ 800e9 / 6.4e6 ≈ 125k tokens/s theoretical upper bound for weight path alone; real tg is far lower due to parallelism/latency/launch, and because attention + overhead add time. Use perf to calibrate BW_eff and per‑op weights actually touched (this sketch intentionally overestimates to be conservative on identifying the bottleneck: bandwidth dominates).


## Flags and build knobs that matter

- GGML_CUDA_FORCE_MMQ / GGML_CUDA_FORCE_CUBLAS: force custom kernels vs BLAS.
  - File ref: llama.cpp/ggml/src/ggml-cuda/mmq.cu:236
- GGML_CUDA_NO_FA: disable FlashAttention (for baseline comparisons).
  - File ref: llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt:92
- HIP specifics: GGML_HIP_ROCWMMA_FATTN (enable WMMA via rocWMMA), GGML_HIP_NO_MMQ_MFMA (disable MFMA path), GGML_HIP_GRAPHS/VMM knobs.
  - File ref: llama.cpp/ggml/src/ggml-hip/CMakeLists.txt:103


## What to optimize next (high‑leverage ideas)

- Improve memory residency/reuse of weight tiles in mmq for batch>1 and grouped rows; for batch=1 inference we are bandwidth‑bound.
- RDNA2/RDNA3 attention: invest in vector/tile FA kernels and shared‑mem tiling to close the gap to WMMA on NVIDIA/CDNA.
- KV cache format tuning: FP16 vs Q8_0 tradeoffs (memory vs attention FP16 compute). Long‑context runs can become attention‑bound; choose KV precision accordingly.
- Stream‑K decomposition thresholds (already used on NVIDIA/CDNA) to improve occupancy on tall/wide GEMMs.


---

If you want, I can add a small script to estimate tg given (d, df, L, S, quant bytes, BW_eff, FP16 TFLOPs) and validate against your GPUs using the perf logs.


## Per‑architecture tables: components, precision, method, library, and share of calc

Notes

- Shares are indicative for LLaMA‑style, weight‑only INT8 export, d≈4096, df≈11008, L≈32.
- “Share (tg)” corresponds to S≈2k; “Share (pp)” corresponds to S≈8k. Shares reflect MACs, not time; time shares vary by arch and kernel efficiency.
- Library “Custom” means llama.cpp ggml kernels (mmq/FA/tile/vec). WMMA refers to nvcuda::wmma (NVIDIA) or rocWMMA (AMD).

NVIDIA Pascal (sm_60/61)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16 storage, F32 accum | elementwise | Custom | Small, memory bound |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | dp4a | Custom (mmq dp4a) | No INT8 Tensor Cores; sm_61 needed for dp4a |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | tile/vector GEMV/GEMM | Custom | No WMMA/MMA; FA uses non‑tensor path |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom | Memory bound |

NVIDIA Volta (sm_70)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16/F32 | elementwise | Custom |  |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | dp4a | Custom (mmq dp4a) | INT8 via dp4a |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | WMMA | WMMA (nvcuda) | Legacy WMMA layout kept for Volta |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom |  |

NVIDIA Turing/Ampere/Ada/Hopper (sm_75/80/86/89/90)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16/F32 | elementwise | Custom |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | MMA int8 (mma.sync); cp.async on Ampere+ | Custom (mmq MMA) | Strong INT8 Tensor Cores; Hopper FP8 unused here |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | MMA FP16 | Custom FA (mma.sync) | High FA perf across these gens |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom |

AMD VEGA20 (GCN)/RDNA2 (gfx906/gfx10.3)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16/F32 | elementwise | Custom | Wave64 on VEGA20; Wave32 on RDNA2 |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | dp4a | Custom (mmq dp4a) | No WMMA on either |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | tile/vector | Custom | Attention non‑tensor; slower kernels |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom |

AMD RDNA3/RDNA4 (gfx11.x/gfx12.x)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16/F32 | elementwise | Custom |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | dp4a | Custom (mmq dp4a) | INT8 compute ~ FP16; bytes dominate |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | WMMA FP16 | WMMA (rocWMMA) | Enable GGML_HIP_ROCWMMA_FATTN; RDNA4 requires rocWMMA ≥ 2.0 |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom |

AMD CDNA1/2/3 (MI100/MI200/MI300)

| Component | Share (tg) | Share (pp) | Precision | Method | Library | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding/Norm/Other | 1–3% | 1–3% | FP16/F32 | elementwise | Custom |
| Projections + MLP | 85–92% | 65–75% | INT8×FP16→INT32→F32 | MFMA int8 | Custom (mmq MFMA) | Strong INT8 MFMA across CDNA |
| Attention QK + PV | 7–12% | 25–35% | FP16→F32 accum | WMMA FP16 | WMMA (rocWMMA) | WMMA availability/version‑dependent; fallback to vector/tile when needed |
| Softmax | <1% | 1–2% | F32 | reduce/exp | Custom |

Unquantized variants

- If weights are FP16/BF16, Projections/MLP rows become FP16 GEMM; library can be Custom (mmf kernels) or cuBLAS/hipBLAS depending on shape/flags.
- Attention precision/method remains as above; FP16 path dominates at large S for pp.
