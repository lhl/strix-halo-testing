# Strix Halo LLM Benchmark Results

All testing was done on pre-production [Framework Desktop](https://frame.work/desktop) systems with an [AMD Ryzen Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) (Strix Halo)/128GB LPDDR5x-8000 configuration. (Thanks Nirav, Alexandru, and co!)

Exact testing/system details are in the results folders, but roughly these are running:
- Close to production BIOS/EC
- Relatively up-to-date kernels: 6.15.5-arch1-1/6.15.6-arch1-1
- Recent TheRock/ROCm-7.0 [nightly builds](https://github.com/ROCm/TheRock/releases/) with Strix Halo (gfx1151) kernels
- Recent [llama.cpp](https://github.com/ggml-org/llama.cpp/) builds (eg [b5863](https://github.com/ggml-org/llama.cpp/tree/b5863) from 2005-07-10)

Just to get a ballpark on the hardware:
- ~215 GB/s max GPU MBW out of a 256 GB/s theoretical (256-bit 8000 MT/s)
- theoretical 59 FP16 TFLOPS (VPOD/WMMA) on RDNA 3.5 (gfx11); effective is *much* lower

## Results

### Prompt Processing (pp) Performance
![PP Performance](summary-results-pp.png)

| Model Name                   | Architecture   |   Weights (B) |   Active (B) | Backend     | Flags      |   pp512 |   tg128 |   Memory (Max MiB) |
|------------------------------|----------------|---------------|--------------|-------------|------------|---------|---------|--------------------|
| Llama 2 7B Q4_0              | Llama 2        |             7 |            7 | Vulkan      |            |   998.0 |  1283.4 |               8646 |
| Shisa V2 8B i1-Q4_K_M        | Llama 3        |             8 |            8 | HIP         | hipBLASLt  |   878.2 |   327.6 |               8344 |
| Qwen 3 30B-A3B UD-Q4_K_XL    | Qwen 3 MoE     |            30 |            3 | Vulkan      | fa=1       |   604.8 |   414.6 |              17874 |
| Mistral Small 3.1 UD-Q4_K_XL | Mistral 3      |            24 |           24 | HIP         | hipBLASLt  |   316.9 |   157.2 |              16090 |
| Hunyuan-A13B UD-Q6_K_XL      | Hunyuan MoE    |            80 |           13 | Vulkan      | fa=1       |   270.5 |   136.7 |              69241 |
| Llama 4 Scout UD-Q4_K_XL     | Llama 4 MoE    |           109 |           17 | HIP         | hipBLASLt  |   264.1 |   141.4 |              60934 |
| Shisa V2 70B i1-Q4_K_M       | Llama 3        |            70 |           70 | HIP rocWMMA |            |    94.7 |    48.6 |              42920 |
| dots1 UD-Q4_K_XL             | dots1 MoE      |           142 |           14 | Vulkan      | fa=1 b=256 |    63.1 |    69.4 |              87564 |

### Text Generation (tg) Performance
![TG Performance](summary-results-tg.png)

| Model Name                   | Architecture   |   Weights (B) |   Active (B) | Backend   | Flags      |   pp512 |   tg128 |   Memory (Max MiB) |
|------------------------------|----------------|---------------|--------------|-----------|------------|---------|---------|--------------------|
| Qwen 3 30B-A3B UD-Q4_K_XL    | Qwen 3 MoE     |            30 |            3 | Vulkan    | b=256      |    70.0 |    72.0 |              17719 |
| Llama 2 7B Q4_0              | Llama 2        |             7 |            7 | Vulkan    | fa=1       |    45.3 |    45.8 |               8080 |
| Shisa V2 8B i1-Q4_K_M        | Llama 3        |             8 |            8 | Vulkan    | fa=1       |    41.9 |    42.0 |               6308 |
| dots1 UD-Q4_K_XL             | dots1 MoE      |           142 |           14 | Vulkan    | fa=1 b=256 |    20.4 |    20.6 |              87557 |
| Llama 4 Scout UD-Q4_K_XL     | Llama 4 MoE    |           109 |           17 | Vulkan    | fa=1 b=256 |    19.2 |    19.3 |              60598 |
| Hunyuan-A13B UD-Q6_K_XL      | Hunyuan MoE    |            80 |           13 | Vulkan    | fa=1 b=256 |    17.1 |    17.1 |              69064 |
| Mistral Small 3.1 UD-Q4_K_XL | Mistral 3      |            24 |           24 | Vulkan    | fa=1       |    14.3 |    14.3 |              15755 |
| Shisa V2 70B i1-Q4_K_M       | Llama 3        |            70 |           70 | Vulkan    | fa=1       |     5.0 |     5.0 |              42583 |

## Testing Notes
The best overall backend and flags were chosen for each model family tested. You can see that often times the best backend for prefill vs token generation differ. Full results for each model (including the pp/tg graphs for different context lengths for all tested backend variations) are available for review in their respective folders as which backend is the best performing will depend on your exact use-case.

There's a lot of performance still on the table when it comes to pp especially. Since these results should be close to optimal for when they were tested, I might add dates to the table  (adding kernel, ROCm, and llama.cpp build#'s might be a bit much).
