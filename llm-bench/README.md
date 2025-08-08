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

## Setup

### GPU Memory
For maximum performance, you should `amd_iommu=off` in your kernel - on memtest_vulkan, this translates to about 6% faster memory reads, although llama.cpp performance tends to be smaller (~2% or less for tg). Still, free performance is free performance. Note, `iommu=pt` does not give any speed benefit.

The other thing you want to do is to set the the GPU memory settings. Here's how to allocate 120GB of memory as GTT GPU memory. Set your GPU memory in BIOS (GART) to the minimum 512MB buffer. In Linux, create a conf in your `/etc/modprobe.d/` (like `/etc/modprobe.d/amdgpu_llm_optimized.conf`):

```
# Maximize GTT for LLM usage on 128GB UMA system
options amdgpu gttsize=120000
options ttm pages_limit=31457280
options ttm page_pool_size=15728640
```

The Translation Table Maps (TTM) is the memory management subsystem that handles GPU memory allocation. `pages_limit` sets the maximum number or 4KiB pages that can be used for GPU memory. `page_pool_size` pre-caches/allocates the memory for usage by the GPU. This will not be available for your system. You probably don't want to set it too high if you will be using the system for other purposes/need memory.

#### tuned
You can improve further improve memory performance with `tuned`:


```bash
paru -S tuned
sudo systemctl enable --now tuned
tuned-adm list
# - accelerator-performance     - Throughput performance based tuning with disabled higher latency STOP states
sudo tuned-adm profile accelerator-performance
tuned-adm active                                                                                                                                                                                         (base)
# Current active profile: accelerator-performance
```

Before:
```
11 iteration. Passed  5.3863 seconds  written:  580.0GB 227.8GB/sec        checked:  609.0GB 214.4GB/sec
```

After:
```
11 iteration. Passed  5.2309 seconds  written:  580.0GB 234.4GB/sec        checked:  609.0GB 221.0GB/sec
```

You might also want to compare memory bandwidth between `AMD_VULKAN_ICD=RADV` and not, different systems seem to behave differently wrt to MBW between these two, which is... curious, to say the least.


### Vulkan
There are multiple [Vulkan libraries](https://wiki.archlinux.org/title/Vulkan) available for AMD and I recommend installing at least two of them. If you're using Arch I recommend:

```
paru -S vulkan-radeon amdvlk vulkan-headers vulkan-tools
```

`amdvlk` (AMDVLK Open) will be used by default and in general seems to be faster than `vulkan-radeon` (Mesa RADV). When both are installed, AMDVLK is used by default, but you can use this env variable: `AMD_VULKAN_ICD=RADV` to use Mesa RADV to test.

I've seen anywhere from no difference to 2X difference in pp performance (AMDVLK always seems to be faster) in limited testing although this may change depending on updates to the libraries. I don't swap Vulkan libs on my current sweeps but it's something to consider in the future...

#### 2025-08-27: tuned and Vulkan implementations
So, a couple interesting discoveries. 

- On my CachyOS system (6.16.0-mainline kernel), Mesa RADV (`vulkan-radeon 1:25.2.0-1`) has slighty faster MBW access and hence higher `tg` vs AMDVLK (`amdvlk 2025.Q2.1-1.1`), but it is flipped on an Arch system running the same kernel. Someone also reported at long context that Mesa RADV was more efficient (eg, 10K+ context) but I haven't tested.

- While there is a slight MBW boost when changing the `tuned` profile `accelerator-performance`, it has minimal impact on `tg`. *However*, it seems to give a significant a 5-8% boost on `pp`!


Here is what AMDVLK looks like before and after (`build: 7ad67ba9 (6111)`):

```
❯ build/bin/llama-bench --mmap 0 -fa 1 -m /models/gguf/llama-2-7b.Q4_0.gguf
```

| model                          |       size |     params | backend    | ngl | fa | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ---: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |  1 |    0 |           pp512 |       1216.56 ± 6.83 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |  1 |    0 |           tg128 |         46.68 ± 0.23 |

| model                          |       size |     params | backend    | ngl | fa | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ---: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |  1 |    0 |           pp512 |       1314.00 ± 2.81 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |  1 |    0 |           tg128 |         46.90 ± 0.03 |


And with Mesa RADV:

```
❯ AMD_VULKAN_ICD=RADV build/bin/llama-bench --mmap 0 -fa 1 -m /models/gguf/llama-2-7b.Q4_0.gguf
```

| model                          |       size |     params | backend    | ngl | fa | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ---: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |  1 |    0 |           pp512 |        882.83 ± 4.53 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |  1 |    0 |           tg128 |         51.56 ± 0.18 |

| model                          |       size |     params | backend    | ngl | fa | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ---: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |  1 |    0 |           pp512 |        931.93 ± 1.98 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan     |  99 |  1 |    0 |           tg128 |         51.88 ± 0.14 |


### ROCm
Instead of using the [ROCm AUR packages](https://wiki.archlinux.org/title/GPGPU#ROCm) it's probably best for now to be using the [latest releases from TheRock](https://github.com/ROCm/TheRock/releases). You can simply make a folder (like `/opt/rocm`) and untar a `gfx1151` release in that.

```
# See latest nightlies
wget https://github.com/ROCm/TheRock/releases/download/nightly-tarball/therock-dist-linux-gfx1151-7.0.0rc20250714.tar.gz
mkdir rocm7.0
cd rocm7.0
tar xvf ../therock-dist-linux-gfx1151-7.0.0rc20250714.tar.gz
```

- You might want to install a gfx110x version first if you want gfx1100 kernels to test as well

- You can choose wherever you want to put it and refer to [rocm-therock-env.sh](../rocm-therock-env.sh) for how to load the appropriate environment variables

- The HIP backend is super crashy on many model architectures due to firmware/driver/ROCm problems? See: https://github.com/ROCm/ROCm/issues/5151

### llama.cpp
You can reference the `[llama-cpp-bencher.py](llama-cpp-bencher.py)` directly for more info, but a few notes:

- There is an [update-llama.cpp.sh](update-llama.cpp.sh) convenience script, but refer to the [official llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for the latest/most authoritative documentation
  - When compiling the ROCm/HIP backend, you may need to tweak `ggml/src/ggml-cuda/vendors/hip.h` if you get a compilation error - The #ifdef versioning used is wrong (deprecations were with 6.5 on) - you might also just need/want to copy the fixed macros into the else statement...
- For the ROCm backend, generally using the hipblaslt libs will be faster for pp (up to 2-3X in some cases!), using the `ROCBLAS_USE_HIPBLASLT=1` environment variable and I'd generally recommend trying that first
- As mentioned, in my testing the AMDVLK Vulkan implementation seems to always be faster, although you can use `AMD_VULKAN_ICD=RADV` if necessary to use the Mesa RADV Vulkan if you have both installed
- Be sure to disable mmap, eg `--mmap 0` for `llama-bench` or `--no-mmap` for `llama-cli` or `llama-server` otherwise you may incur extreme model loading speed penalties with the ROCm backend if exceeding 50% of the available system memory

Sometimes the pp512/tg128 don't tell the whole story. If you're deciding on the optimal backend to use for a specific model, you may want to add pp4096/tg2048 just to see how the drop-off is at higher context.

#### rocWMMA
If you are using ROCm 6.5+ then llama.cppw/ rocWMMA will likely not work out of the box:

First make sure you have the latest rocWMMA headers installed:
```
./build-rocwmma.sh
```

Then you can apply patches:
```
./apply-rocwmma-fix.sh ~/llama.cpp/llama.cpp-rocwmma
```

Now you should be able to compile as normal:
```
cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=ON  && cmake --build build --config Release -j32
```


## Results

### Prompt Processing (pp) Performance
![PP Performance](summary-results-pp.png)

| Model Name                   | Architecture   |   Weights (B) |   Active (B) | Backend       | Flags          |   pp512 |   tg128 |   Memory (Max MiB) |
|------------------------------|----------------|---------------|--------------|---------------|----------------|---------|---------|--------------------|
| Llama 2 7B Q4_0              | Llama 2        |             7 |            7 | Vulkan AMDVLK | fa=1           |  1294.1 |    47.7 |               4281 |
| OpenAI gpt-oss 20B MXFP4     | gpt-oss        |            21 |            4 | HIP           | hipBLASLt      |  1196.2 |    46.9 |              12958 |
| Llama 2 7B Q4_K_M            | Llama 2        |             7 |            7 | HIP rocWMMA   | fa=1 hipBLASLt |  1083.0 |    41.3 |               4720 |
| Shisa V2 8B i1-Q4_K_M        | Llama 3        |             8 |            8 | HIP           | hipBLASLt      |   878.2 |    37.2 |               5308 |
| Qwen 3 30B-A3B UD-Q4_K_XL    | Qwen 3 MoE     |            30 |            3 | HIP rocWMMA   | fa=1 hipBLASLt |   669.4 |    58.5 |              17533 |
| OpenAI gpt-oss 120B MXFP4    | gpt-oss        |           117 |            5 | Vulkan AMDVLK |                |   430.6 |    33.7 |              64187 |
| Mistral Small 3.1 UD-Q4_K_XL | Mistral 3      |            24 |           24 | HIP rocWMMA   | fa=1 hipBLASLt |   368.5 |    13.8 |              14638 |
| Gemma 3 27B UD-Q4_K_XL       | Gemma 3        |            27 |           27 | HIP           | hipBLASLt      |   302.2 |    10.7 |              17542 |
| Hunyuan-A13B UD-Q6_K_XL      | Hunyuan MoE    |            80 |           13 | Vulkan AMDVLK | fa=1           |   296.6 |    18.1 |              69179 |
| Llama 4 Scout UD-Q4_K_XL     | Llama 4 MoE    |           109 |           17 | HIP           | hipBLASLt      |   277.4 |    17.6 |              59720 |
| Qwen 3 32B Q8_0              | Qwen 3         |            32 |           32 | HIP           | hipBLASLt      |   226.1 |     6.4 |              33683 |
| dots1 UD-Q4_K_XL             | dots1 MoE      |           142 |           14 | Vulkan AMDVLK | fa=1           |   182.0 |    22.1 |              84082 |
| GLM 4.5 Air UD-Q4_K_XL       | GLM 4.5        |           106 |           12 | Vulkan AMDVLK | fa=1           |   178.5 |    22.6 |              68419 |
| Qwen 3 235B-A22B UD-Q3_K_XL  | Qwen 3 MoE     |           235 |           22 | HIP           | hipBLASLt      |   117.1 |    12.9 |              99950 |
| Shisa V2 70B i1-Q4_K_M       | Llama 3        |            70 |           70 | HIP rocWMMA   | hipBLASLt      |    94.7 |     4.5 |              41522 |

### Text Generation (tg) Performance
![TG Performance](summary-results-tg.png)

| Model Name                   | Architecture   |   Weights (B) |   Active (B) | Backend       | Flags      |   pp512 |   tg128 |   Memory (Max MiB) |
|------------------------------|----------------|---------------|--------------|---------------|------------|---------|---------|--------------------|
| Qwen 3 30B-A3B UD-Q4_K_XL    | Qwen 3 MoE     |            30 |            3 | Vulkan AMDVLK | b=256      |   645.5 |    78.0 |              17377 |
| Llama 2 7B Q4_0              | Llama 2        |             7 |            7 | Vulkan RADV   | fa=1       |   924.9 |    51.2 |               4276 |
| Llama 2 7B Q4_K_M            | Llama 2        |             7 |            7 | Vulkan AMDVLK | fa=1       |   787.6 |    48.7 |               4463 |
| OpenAI gpt-oss 20B MXFP4     | gpt-oss        |            21 |            4 | Vulkan AMDVLK | b=256      |   956.5 |    47.1 |              14690 |
| Shisa V2 8B i1-Q4_K_M        | Llama 3        |             8 |            8 | Vulkan AMDVLK | fa=1       |   614.2 |    42.0 |               5333 |
| OpenAI gpt-oss 120B MXFP4    | gpt-oss        |           117 |            5 | Vulkan AMDVLK | b=256      |   386.8 |    33.7 |              63972 |
| GLM 4.5 Air UD-Q4_K_XL       | GLM 4.5        |           106 |           12 | Vulkan RADV   | fa=1       |   125.4 |    23.4 |              68365 |
| dots1 UD-Q4_K_XL             | dots1 MoE      |           142 |           14 | Vulkan AMDVLK | fa=1 b=256 |   139.1 |    22.1 |              83917 |
| Llama 4 Scout UD-Q4_K_XL     | Llama 4 MoE    |           109 |           17 | Vulkan AMDVLK | fa=1 b=256 |   157.8 |    19.4 |              59917 |
| Hunyuan-A13B UD-Q6_K_XL      | Hunyuan MoE    |            80 |           13 | Vulkan AMDVLK | fa=1 b=256 |   244.8 |    18.1 |              69006 |
| Qwen 3 235B-A22B UD-Q3_K_XL  | Qwen 3 MoE     |           235 |           22 | Vulkan AMDVLK | fa=1       |   109.5 |    15.1 |             100446 |
| Mistral Small 3.1 UD-Q4_K_XL | Mistral 3      |            24 |           24 | Vulkan AMDVLK | fa=1       |   203.3 |    14.4 |              14540 |
| Gemma 3 27B UD-Q4_K_XL       | Gemma 3        |            27 |           27 | Vulkan AMDVLK | fa=1       |   114.7 |    11.8 |              18123 |
| Qwen 3 32B Q8_0              | Qwen 3         |            32 |           32 | Vulkan AMDVLK | fa=1       |   101.8 |     6.4 |              33886 |
| Shisa V2 70B i1-Q4_K_M       | Llama 3        |            70 |           70 | Vulkan AMDVLK | fa=1       |    26.4 |     5.0 |              41456 |


## Testing Notes
The best overall backend and flags were chosen for each model family tested. You can see that often times the best backend for prefill vs token generation differ. Full results for each model (including the pp/tg graphs for different context lengths for all tested backend variations) are available for review in their respective folders as which backend is the best performing will depend on your exact use-case.

There's a lot of performance still on the table when it comes to pp especially. Since these results should be close to optimal for when they were tested, I might add dates to the table  (adding kernel, ROCm, and llama.cpp build#'s might be a bit much).

## Additional Resources

- [Strix Halo HomeLab](https://strixhalo-homelab.d7.wtf/) - there is a wiki and Discord for those looking to get dig deeper into many technical aspects of running these Strix Halo machines
- [AMD Strix Halo Llama.cpp Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) - scripts for easily running docker containers with precompiled llama.cpp backends, and some additional testing notes
- [Beowulf AI Cluster](https://github.com/geerlingguy/beowulf-ai-cluster) - for clustering, Jeff Geerling has been doing a bunch of work including writing ansible scripts for deployment; be sure to check the issues in the repo and his [Framework Desktop review](https://github.com/geerlingguy/sbc-reviews/issues/80)
