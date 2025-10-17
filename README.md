# strix-halo-testing

This repository contains my testing and development for the AMD Strix Halo (Ryzen AI Max+ 395) APU (particularly the gfx1151 RNDA 3.5 GPU).

All this work was done on a pre-production Framework Desktop / remote Framework cluster courtesy of Framework (thanks guys!) with the goal of seeing if we could get Strix Halo actually useful/usable for local AI, and for getting some documentation along the lines of my  [RDNA3 AI/ML doc](https://llm-tracker.info/howto/AMD-GPUs).

My WIP Documentation here: https://llm-tracker.info/_TOORG/Strix-Halo but since the release of the Framework Desktop, I've been focusing my documentation efforts on the [AI section of the Strix Halo HomeLab Wiki](https://strixhalo-homelab.d7.wtf/AI/AI-Capabilities-Overview). That should be considered the most up-to-date starting point and covers a summary of the technical capabilities, basic system setup and tweaks, and well as some docs on llama.cpp and vLLM setup.

## ROCm environment scripts

I have some legacy `rocm-env.sh` scripts that may be useful, but for mamba/env setup that leverages the latest pip-based [ROCm/TheRock RELEASE](https://github.com/ROCm/TheRock/blob/main/RELEASES.md), see my [torch-therock/00-setup-env.sh](https://github.com/lhl/strix-halo-testing/blob/main/torch-therock/00-setup-env.sh) script, which leverages the `rocm-sdk` tool to generate the proper ROCM paths.

## Most Useful Parts

This is a "working" repo, but there are a few things that are most worth looking at:

- `hardware-test` - if you're looking for raw memory-bandwidth and performance testing, this is a good folder to look at
- `llm-bench` - I ran a wide range of performance sweeps (which includes longer context across multiple llama.cpp backends) to characterize LLM performance. For some more up-to-date (pp512/tg128) numbers, check out [kyuz0's Interactive Viewer](https://kyuz0.github.io/amd-strix-halo-toolboxes/) - I also have a [llama.cpp performance](https://strixhalo-homelab.d7.wtf/AI/llamacpp-performance) doc that goes over more of how to test, and things to consider/look out for
- `rpc-test` - basic testing of how to use the llama.cpp RPC backend for clustering. If you have an interest in this, you'll probably want to [check out Jeff Geerling's work](https://github.com/geerlingguy/ollama-benchmark/issues/21) on the topic
- `torch-therock` - as of 2025-10-15 there is no AOTriton for PyTorch (and hence no Flash Attention!) being built automatically (see[ROCm/TheRock #1408](https://github.com/ROCm/TheRock/issues/1408)), but this is the script I use to build my own PyTorch + AOTriton
- `vllm` - I use this to be able to build my own vLLM. AFAIK, this was a first, but is not for the faint of heart, and kyuz0 and others have since used this approach to build easier to use versions. If you're not already knee-deep in monkey-patching/struggling w/ vLLM builds, you'll probably want to check out this [AMD Strix Halo — vLLM Toolbox/Container (gfx1151, PyTorch + AOTriton)](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes) instead!

## Strix Halo Notes

- I've done a fair amount of [posting on the Framework Forums](https://community.frame.work/u/lhl/activity) on some Strix Halo details that may be useful (of course no one has time for reading all that, but just linking in case)

## AMD Strix Halo vs Nvidia DGX Spark

I don't have an Nvidia DGX Spark but I may get SSH access to one and may do a bit of poking around. If you think of the Strix Halo GPU as a Radeon RX 7600 XT with 128GB of LPDDR5X, you can think about the Spark as a (very low power) RTX 5070 with 128GB of LPDDR5X. Beyond that, the Spark has a [very slick getting started experience](https://build.nvidia.com/spark) and a [huge number of Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) that along with its CUDA support makes it well suited for those just getting started with AI/ML development.

That being said, while the Spark has a much more mature software ecosystem and a lot more compute that gives it both a significant prefill/prompt processing advantage for LLMs and makes it much more suitable for image, audio and video generation, for token generation/decode on big LLMs, the two devices are essentially neck-and-neck at short context with Vulkan, though Spark does pull ahead at longer context and with ROCm backends.

ggeranov ran a fair amount of [llama.cpp performance sweeps](https://github.com/ggml-org/llama.cpp/discussions/16578) on launch (perf actually has since improved/been updated). I was curious and ran some comparisons vs my Strix Halo (Framework Desktop, Arch 6.17.0-1-mainline, all optimizations (amd_iommu, tuned) set properly).  I only tested against his gpt-oss-120b tests (this is the ggml-org one, so Q8/MXFP4).

This was tested with TheRock/ROCm nightly (7.10.0a20251014) and the latest Vulkan drivers (RADV 25.2.4-2, AMDVLK 2025.Q2.1-1) and I've picked the faster overall numbers for Vulkan (AMDVLK atm) and ROCm (regular hipblas w/ rocWMMA). llama.cpp build is 6763, almost the same as ggeranov's build (6767).

Token generation/decode performance is essentially even at short context with Vulkan (Spark +0.2% at 2K context), while Spark pulls ahead with ROCm (+11.7% at 2K context, +108.5% at 32K context). The CUDA backend drops less as context increases vs either Vulkan or ROCm (Vulkan does better than ROCm as context increases - at 32K context, Vulkan `tg` is 2X ROCm!). For prompt processing/prefill, Strix Halo gets crushed on `pp` tests. In the best case (ROCm), Strix Halo starts off over 2X slower and by 32K gets to 5X slower, dropping off over twice as fast in performance as context extends.

## Vulkan AMDVLK

| Test          |     DGX |   STXH |       % |
| ------------- | ------: | -----: | ------: |
| pp2048        | 1689.47 | 729.59 | +131.6% |
| pp2048@d4096  | 1733.41 | 563.30 | +207.7% |
| pp2048@d8192  | 1705.93 | 424.52 | +301.8% |
| pp2048@d16384 | 1514.78 | 260.18 | +482.2% |
| pp2048@d32768 | 1221.23 | 152.56 | +700.5% |

| Test        |   DGX |  STXH |      % |
| ----------- | ----: | ----: | -----: |
| tg32        | 52.87 | 52.74 |  +0.2% |
| tg32@d4096  | 51.02 | 49.49 |  +3.1% |
| tg32@d8192  | 48.46 | 46.94 |  +3.2% |
| tg32@d16384 | 44.78 | 42.85 |  +4.5% |
| tg32@d32768 | 38.76 | 36.31 |  +6.7% |

## ROCm w/ rocWMMA

| Test          |     DGX |   STXH |       % |
| ------------- | ------: | -----: | ------: |
| pp2048        | 1689.47 | 735.77 | +129.6% |
| pp2048@d4096  | 1733.41 | 621.88 | +178.7% |
| pp2048@d8192  | 1705.93 | 535.84 | +218.4% |
| pp2048@d16384 | 1514.78 | 384.69 | +293.8% |
| pp2048@d32768 | 1221.23 | 242.19 | +404.2% |

| Test        |   DGX |  STXH |      % |
| ----------- | ----: | ----: | -----: |
| tg32        | 52.87 | 47.35 | +11.7% |
| tg32@d4096  | 51.02 | 40.77 | +25.1% |
| tg32@d8192  | 48.46 | 34.50 | +40.5% |
| tg32@d16384 | 44.78 | 26.86 | +66.7% |
| tg32@d32768 | 38.76 | 18.59 | +108.5% |
