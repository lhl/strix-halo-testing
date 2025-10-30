# Improving llama.cpp HIP Backend Performance

- Source: https://github.com/lhl/llama.cpp/tree/rocm-wmma-tune
  - Changes: https://github.com/ggml-org/llama.cpp/compare/master...lhl:llama.cpp:rocm-wmma-tune
- PR & Discussion: https://github.com/ggml-org/llama.cpp/pull/16827

Recently some people on the [Strix Halo HomeLab](https://strixhalo-homelab.d7.wtf/) Discord noticed that the HIP rocWMMA backend which [used to have some big pp performance advantages](https://github.com/lemonade-sdk/llamacpp-rocm/issues/7) actually recently was on par for `pp512`/`tg128` w/ the standard HIP path, and *also* performed notably worse than the HIP path (especially for decode/token generation) as context got longer.

After some research and analysis of [how ggml-cuda works](llama-cpp-cuda-hip.md), I discovered the rocWMMA path was actually an old deprecated path. I gave it a quick poke to see if it could be easily fixed.

The approach was to see if some minimal HIP-isolated changes to improve performance much. The results showed that actually, there were huge optimizations available and that it'd be relatively clean to patch in improvements.

The big things discovered:
- There were a few parameters on the WMMA path that could be dynamically calculated to dramatically increase HIP occupancy and reduce LDS footprint - these show across the board prefill improvements that increase with longer-context, up to 50-100%+ 64K and increasing
- The poor long-context decode is because it always falls back to the VEC, not using TILE kernels when appropriate. There was a second issue of some weird TILE constraints (and no guards or fallbacks) also.

Once these were fixed, you could get a huge improvement for prefill over both the prior HIP and WMMA implementation at all contexts (but especially at long context), and then match the HIP performance for tg.

## Building llama.cpp
```
# HIP
cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 && cmake --build build --config Release -j32

# rocWMMA
 cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=ON && cmake --build build --config Release -j32

# use rocWMMA build for feature branch
```

## Benchmarking
```
# Get the HIP & WMMA baseline
./run-llama-bench.sh --model <path/to/test/GGUF> --tag <model-name>

# Get new WMMA results 
llama.cpp/build/bin/llama-bench -m <path/to/test/GGUF> -fa 1 -d 0,4096,8192,16384,65536 -mmp 0 -o jsonl > <model-name.your-label.jsonl>

# Compare
python analyze-results.py <model-name.your-label.jsonl>
```

Llama 3.2 1B Q4_K_M was used for basic performance profiling during development (about 3min/full run for doing full sweep).

### Llama 3.2 1B Q4_K_M

#### Previous rocWMMA vs HIP
 
Prefill (pp)

| model                  |       size |   params | test           |     HIP |    WMMA |      Δ% |
|------------------------|------------|----------|----------------|--------:|--------:|--------:|
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512          | 4703.28 | 4884.42 |   3.85% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d1024  | 4076.03 | 4204.81 |   3.16% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d4096  | 2936.89 | 2959.54 |   0.77% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d16384 | 1350.48 | 1265.62 |  -6.28% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d65536 |  424.76 |  360.24 | -15.19% |

Decode (tg)

| model                  |       size |   params | test           |    HIP |   WMMA |      Δ% |
|------------------------|------------|----------|----------------|-------:|-------:|--------:|
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128          | 195.65 | 193.01 |  -1.35% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d1024  | 188.79 |  182.6 |  -3.28% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d4096  | 173.36 | 143.51 | -17.22% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d16384 | 126.86 |  87.53 | -31.01% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d65536 |  64.62 |  27.35 | -57.68% |


#### My rocWMMA vs HIP

Prefill (pp)

| model                  |       size |   params | test           |     HIP |   lhl-tune-tile |     Δ% |
|------------------------|------------|----------|----------------|--------:|----------------:|-------:|
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512          | 4703.28 |         4970.14 |  5.67% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d1024  | 4076.03 |         4575.18 | 12.25% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d4096  | 2936.89 |         3788.92 | 29.01% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d16384 | 1350.48 |         2064.78 | 52.89% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d65536 |  424.76 |          706.46 | 66.32% |

Decode (tg)

| model                  |       size |   params | test           |    HIP |   lhl-tune-tile |     Δ% |
|------------------------|------------|----------|----------------|-------:|----------------:|-------:|
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128          | 195.65 |          195.59 | -0.03% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d1024  | 188.79 |          188.84 |  0.03% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d4096  | 173.36 |          173.28 | -0.05% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d16384 | 126.86 |          127.01 |  0.12% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d65536 |  64.62 |           64.55 | -0.10% |

#### My rocWMMA vs Previous rocWMMA

Prefill (pp)

| model                  |       size |   params | test           |   default-rocwmma |   lhl-tune-tile |     Δ% |
|------------------------|------------|----------|----------------|------------------:|----------------:|-------:|
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512          |           4884.42 |         4970.14 |  1.75% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d1024  |           4204.81 |         4575.18 |  8.81% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d4096  |           2959.54 |         3788.92 | 28.02% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d16384 |           1265.62 |         2064.78 | 63.14% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | pp512 @ d65536 |            360.24 |          706.46 | 96.11% |

Decode (tg)

| model                  |       size |   params | test           |   default-rocwmma |   lhl-tune-tile |      Δ% |
|------------------------|------------|----------|----------------|------------------:|----------------:|--------:|
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128          |            193.01 |          195.59 |   1.34% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d1024  |             182.6 |          188.84 |   3.42% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d4096  |            143.51 |          173.28 |  20.74% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d16384 |             87.53 |          127.01 |  45.11% |
| llama 1B Q4_K - Medium | 762.81 MiB |   1.24 B | tg128 @ d65536 |             27.35 |           64.55 | 136.06% |


### gpt-oss-20b F16/MXFP4

#### Previous rocWMMA vs HIP

Prefill (pp)
| model           |         size |   params | test           |     HIP |    WMMA |      Δ% |
|-----------------|--------------|----------|----------------|--------:|--------:|--------:|
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512          | 1472.01 | 1513.79 |   2.84% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d1024  | 1387.58 | 1417.45 |   2.15% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d4096  | 1175.72 | 1205.37 |   2.52% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d16384 |   713.9 |  669.77 |  -6.18% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d65536 |  277.58 |  227.24 | -18.14% |

Decode (tg)
| model           |         size |   params | test           |   HIP |   WMMA |      Δ% |
|-----------------|--------------|----------|----------------|------:|-------:|--------:|
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128          | 49.92 |  50.23 |   0.61% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d1024  | 49.27 |  48.65 |  -1.26% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d4096  | 48.15 |  45.11 |  -6.32% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d16384 | 44.38 |  32.91 | -25.85% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d65536 | 34.76 |  14.63 | -57.92% |

#### My rocWMMA vs HIP

Prefill (pp)

| model           |         size |   params | test           |     HIP |   lhl-tune-tile |     Δ% |
|-----------------|--------------|----------|----------------|--------:|----------------:|-------:|
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512          | 1472.01 |         1495.97 |  1.63% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d1024  | 1387.58 |         1456.15 |  4.94% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d4096  | 1175.72 |         1347.75 | 14.63% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d16384 |   713.9 |          962.98 | 34.89% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d65536 |  277.58 |          426.81 | 53.76% |

Decode (tg)

| model           |         size |   params | test           |   HIP |   lhl-tune-tile |     Δ% |
|-----------------|--------------|----------|----------------|------:|----------------:|-------:|
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128          | 49.92 |            49.9 | -0.04% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d1024  | 49.27 |           49.21 | -0.11% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d4096  | 48.15 |           48.05 | -0.20% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d16384 | 44.38 |           44.34 | -0.11% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d65536 | 34.76 |           34.77 |  0.03% |

#### My rocWMMA vs Previous rocWMMA

Prefill (pp)

| model           |         size |   params | test           |   default-rocwmma |   lhl-tune-tile |     Δ% |
|-----------------|--------------|----------|----------------|------------------:|----------------:|-------:|
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512          |           1513.79 |         1495.97 | -1.18% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d1024  |           1417.45 |         1456.15 |  2.73% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d4096  |           1205.37 |         1347.75 | 11.81% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d16384 |            669.77 |          962.98 | 43.78% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | pp512 @ d65536 |            227.24 |          426.81 | 87.83% |

Decode (tg)

| model           |         size |   params | test           |   default-rocwmma |   lhl-tune-tile |      Δ% |
|-----------------|--------------|----------|----------------|------------------:|----------------:|--------:|
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128          |             50.23 |            49.9 |  -0.64% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d1024  |             48.65 |           49.21 |   1.16% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d4096  |             45.11 |           48.05 |   6.53% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d16384 |             32.91 |           44.34 |  34.72% |
| gpt-oss 20B F16 | 13141.28 MiB |  20.91 B | tg128 @ d65536 |             14.63 |           34.77 | 137.71% |


## Reference

- [IMPLEMENTATION-better-rocwmma.md](IMPLEMENTATION-better-rocwmma.md)
  - Worklog, etc
- [IMPLEMENTATION-native-rocm.md](IMPLEMENTATION-native-rocm.md)
  - PROPOSED further improvements (unlikely to be done, ggml-cuda maintainer plans to do their own rewrite)
- [llama-cpp-cuda-hip.md](llama-cpp-cuda-hip.md)
  - ggml-cuda reference doc generated from spelunking through code (multiple passes from GPT5 High, Claude Sonnet 4.5 ultrathink, GPT5 Codex High)
- [llama.cpp PR#16827](https://github.com/ggml-org/llama.cpp/pull/16827)
  - pull request rejected since there's a planned rewrite (getting commits upstreamed to ggml-cuda looks like a [full-time job](https://github.com/ggml-org/llama.cpp/pull/14624))
