#!/usr/bin/env python3
"""
Strix Halo vs DGX Spark Benchmark Comparison
=============================================

This script compares LLM inference performance between AMD Ryzen AI Max+ 395
(Strix Halo) and DGX Spark systems using llama.cpp benchmarks.

Data Sources & Methodology
---------------------------

DGX Spark Benchmarks:
  - Source: ggerganov's llama.cpp discussion
  - URL: https://github.com/ggml-org/llama.cpp/discussions/16578
  - Hardware: DGX Spark system

Strix Halo Benchmarks:
  - CPU: AMD Eng Sample: 100-000001243-50_Y (Ryzen AI Max+ 395)
  - GPU: Strix Halo [Radeon Graphics / Radeon 8050S / 8060S] (rev d1)
  - Kernel: 6.17.0-1-mainline (amd_iommu=off)
  - Tuned: accelerator-performance profile
  - Vulkan: AMDVLK 2025.Q2.1-1, RADV (Mesa 25.2.4-2)
  - ROCm: 7.10.0a20251017

Benchmark Command (used for all tests):
  build/bin/llama-bench -fa 1 -d 0,4096,8192,16384,32768 -p 2048 -n 32 -ub 2048 \
    -m /models/gguf/gpt-oss-120b-mxfp4-00001-of-00003.gguf

  Parameters:
    -fa 1               : Flash Attention enabled
    -d <values>         : Context sizes (0, 4096, 8192, 16384, 32768)
    -p 2048             : Prompt size
    -n 32               : Number of tokens to generate
    -ub 2048            : Uniform batch size
    -m <model>          : GPT-OSS-120B MXFP4 quantized model

  llama.cpp build: 81387858f (6792)

Build Configurations
--------------------

ROCm Build:
  cmake -S . -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1151 \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
  && cmake --build build --config Release -j32

Vulkan Build:
  cmake -S . -B build \
    -DGGML_VULKAN=on \
    -DGGML_RPC=ON \
  && cmake --build build --config Release -j32

Usage
-----
  python strix-vs-spark.py                  # Display console output and markdown tables
  python strix-vs-spark.py --update-readme  # Update README.md Performance section

Output
------
The script calculates percentage differences between DGX Spark and Strix Halo
performance, showing how much faster (+) or slower (-) the DGX Spark is compared
to Strix Halo. All percentages in the narrative are calculated dynamically from
the benchmark data, ensuring consistency. Console-formatted and markdown table
outputs are generated, and the Performance Results section can be automatically
updated in README.md with the --update-readme flag.
"""

# New DGX Spark numbers
dgx_data = {
    'pp2048': 1689.47,
    'pp2048@d4096': 1733.41,
    'pp2048@d8192': 1705.93,
    'pp2048@d16384': 1514.78,
    'pp2048@d32768': 1221.23,
    'tg32': 52.87,
    'tg32@d4096': 51.02,
    'tg32@d8192': 48.46,
    'tg32@d16384': 44.78,
    'tg32@d32768': 38.76,
}

# Strix Halo numbers - Vulkan AMDVLK (optimal -ub 512)
stxh_vulkan = {
    'pp2048': 729.10,
    'pp2048@d4096': 562.15,
    'pp2048@d8192': 424.50,
    'pp2048@d16384': 249.68,
    'pp2048@d32768': 137.08,
    'tg32': 50.05,
    'tg32@d4096': 46.11,
    'tg32@d8192': 43.15,
    'tg32@d16384': 38.46,
    'tg32@d32768': 31.54,
}

# Strix Halo numbers - ROCm w/ rocWMMA
stxh_rocm = {
    'pp2048': 1006.65,
    'pp2048@d4096': 790.45,
    'pp2048@d8192': 603.83,
    'pp2048@d16384': 405.53,
    'pp2048@d32768': 223.82,
    'tg32': 46.56,
    'tg32@d4096': 38.25,
    'tg32@d8192': 32.65,
    'tg32@d16384': 25.50,
    'tg32@d32768': 17.82,
}

def calculate_percentage(dgx, stxh):
    """Calculate percentage difference: ((DGX - STXH) / STXH) * 100"""
    return ((dgx - stxh) / stxh) * 100

def format_percentage(pct):
    """Format percentage with proper sign."""
    return f"{pct:+.1f}%"

def generate_readme_section():
    """Generate the complete Performance Results section for README.md"""
    # Calculate key percentages for narrative
    vulkan_tg32 = calculate_percentage(dgx_data['tg32'], stxh_vulkan['tg32'])
    rocm_tg32 = calculate_percentage(dgx_data['tg32'], stxh_rocm['tg32'])
    rocm_tg32_32k = calculate_percentage(dgx_data['tg32@d32768'], stxh_rocm['tg32@d32768'])
    rocm_pp2048 = calculate_percentage(dgx_data['pp2048'], stxh_rocm['pp2048'])
    rocm_pp2048_32k = calculate_percentage(dgx_data['pp2048@d32768'], stxh_rocm['pp2048@d32768'])
    vulkan_pp2048 = calculate_percentage(dgx_data['pp2048'], stxh_vulkan['pp2048'])
    vulkan_pp2048_32k = calculate_percentage(dgx_data['pp2048@d32768'], stxh_vulkan['pp2048@d32768'])

    # Calculate Vulkan vs ROCm ratio at 32K context
    vulkan_rocm_ratio = stxh_vulkan['tg32@d32768'] / stxh_rocm['tg32@d32768']

    # Generate narrative
    narrative = f"""## AMD Strix Halo vs Nvidia DGX Spark

I don't have an Nvidia DGX Spark but I may get SSH access to one and may do a bit of poking around. If you think of the Strix Halo GPU as a Radeon RX 7600 XT with 128GB of LPDDR5X, you can think about the Spark as a (very low power) RTX 5070 with 128GB of LPDDR5X. Beyond that, the Spark has a [very slick getting started experience](https://build.nvidia.com/spark) and a [huge number of Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) that along with its CUDA support makes it well suited for those just getting started with AI/ML development.

That being said, while the Spark has a much more mature software ecosystem and a lot more compute that gives it both a significant prefill/prompt processing advantage for LLMs and makes it much more suitable for image, audio and video generation, for token generation/decode on big LLMs, the two devices are essentially neck-and-neck at short context with Vulkan, though Spark does pull ahead at longer context and with ROCm backends.

ggeranov ran a fair amount of [llama.cpp performance sweeps](https://github.com/ggml-org/llama.cpp/discussions/16578) on launch (perf actually has since improved/been updated). I was curious and ran some comparisons vs my Strix Halo (Framework Desktop, Arch 6.17.0-1-mainline, all optimizations (amd_iommu, tuned) set properly).  I only tested against his gpt-oss-120b tests (this is the ggml-org one, so Q8/MXFP4).

This was tested with TheRock/ROCm nightly (7.10.0a20251017) and the latest Vulkan drivers (RADV 25.2.4-2, AMDVLK 2025.Q2.1-1) and I've picked the faster overall numbers for Vulkan (AMDVLK atm) and ROCm (regular hipblas w/ rocWMMA). llama.cpp build is 6792, almost the same as ggeranov's build (6767).

Token generation/decode performance is essentially even at short context with Vulkan (Spark {format_percentage(vulkan_tg32)} at 2K context), while Spark pulls ahead with ROCm ({format_percentage(rocm_tg32)} at 2K context, {format_percentage(rocm_tg32_32k)} at 32K context). The CUDA backend drops less as context increases vs either Vulkan or ROCm (Vulkan does better than ROCm as context increases - at 32K context, Vulkan `tg` is {vulkan_rocm_ratio:.1f}X ROCm!). For prompt processing/prefill, ROCm performance has improved significantly but Strix Halo still lags behind Spark. With ROCm, Strix Halo starts off {format_percentage(rocm_pp2048)} slower at 2K context and by 32K gets to {format_percentage(rocm_pp2048_32k)} slower, while Vulkan ranges from {format_percentage(vulkan_pp2048)} to {format_percentage(vulkan_pp2048_32k)} slower.

**Note on Vulkan drivers and batch sizes:**
- AMDVLK (shown below) uses optimal `-ub 512` and has better `pp` performance
- RADV uses optimal `-ub 1024` with lower `pp` but `tg` decreases less at depth
- ROCm tested with standard `-ub 2048`

### Vulkan AMDVLK

| Test          |     DGX |   STXH |       % |
| ------------- | ------: | -----: | ------: |"""

    # Add Vulkan pp rows
    for test in ['pp2048', 'pp2048@d4096', 'pp2048@d8192', 'pp2048@d16384', 'pp2048@d32768']:
        dgx = dgx_data[test]
        stxh = stxh_vulkan[test]
        pct = calculate_percentage(dgx, stxh)
        narrative += f"\n| {test:<13} | {dgx:>7.2f} | {stxh:>6.2f} | {format_percentage(pct):>7} |"

    narrative += "\n\n| Test        |   DGX |  STXH |      % |\n| ----------- | ----: | ----: | -----: |"

    # Add Vulkan tg rows
    for test in ['tg32', 'tg32@d4096', 'tg32@d8192', 'tg32@d16384', 'tg32@d32768']:
        dgx = dgx_data[test]
        stxh = stxh_vulkan[test]
        pct = calculate_percentage(dgx, stxh)
        narrative += f"\n| {test:<11} | {dgx:>5.2f} | {stxh:>5.2f} | {format_percentage(pct):>6} |"

    narrative += "\n\n### ROCm w/ rocWMMA\n\n| Test          |     DGX |   STXH |       % |\n| ------------- | ------: | -----: | ------: |"

    # Add ROCm pp rows
    for test in ['pp2048', 'pp2048@d4096', 'pp2048@d8192', 'pp2048@d16384', 'pp2048@d32768']:
        dgx = dgx_data[test]
        stxh = stxh_rocm[test]
        pct = calculate_percentage(dgx, stxh)
        narrative += f"\n| {test:<13} | {dgx:>7.2f} | {stxh:>6.2f} | {format_percentage(pct):>7} |"

    narrative += "\n\n| Test        |   DGX |  STXH |      % |\n| ----------- | ----: | ----: | -----: |"

    # Add ROCm tg rows
    for test in ['tg32', 'tg32@d4096', 'tg32@d8192', 'tg32@d16384', 'tg32@d32768']:
        dgx = dgx_data[test]
        stxh = stxh_rocm[test]
        pct = calculate_percentage(dgx, stxh)
        narrative += f"\n| {test:<11} | {dgx:>5.2f} | {stxh:>5.2f} | {format_percentage(pct):>6} |"

    return narrative

def update_readme(readme_path='README.md'):
    """Update the Performance Results section in README.md"""
    import re

    with open(readme_path, 'r') as f:
        content = f.read()

    # Find the section starting with "## AMD Strix Halo vs Nvidia DGX Spark" to end of file
    pattern = r'## AMD Strix Halo vs Nvidia DGX Spark.*$'

    new_section = generate_readme_section()

    # Replace the section
    updated_content = re.sub(pattern, new_section, content, flags=re.DOTALL)

    with open(readme_path, 'w') as f:
        f.write(updated_content)

    print(f"✓ Updated {readme_path}")

if __name__ == "__main__":
    import sys

    # Check if user wants to update README
    if len(sys.argv) > 1 and sys.argv[1] == '--update-readme':
        update_readme()
        sys.exit(0)

print("=" * 60)
print("VULKAN AMDVLK")
print("=" * 60)

print("\nPrompt Processing (pp) Tests:")
print(f"{'Test':<15} {'DGX':>8} {'STXH':>8} {'%':>10}")
print("-" * 45)
for test in ['pp2048', 'pp2048@d4096', 'pp2048@d8192', 'pp2048@d16384', 'pp2048@d32768']:
    dgx = dgx_data[test]
    stxh = stxh_vulkan[test]
    pct = calculate_percentage(dgx, stxh)
    print(f"{test:<15} {dgx:>8.2f} {stxh:>8.2f} {format_percentage(pct):>10}")

print("\nToken Generation (tg) Tests:")
print(f"{'Test':<15} {'DGX':>8} {'STXH':>8} {'%':>10}")
print("-" * 45)
for test in ['tg32', 'tg32@d4096', 'tg32@d8192', 'tg32@d16384', 'tg32@d32768']:
    dgx = dgx_data[test]
    stxh = stxh_vulkan[test]
    pct = calculate_percentage(dgx, stxh)
    print(f"{test:<15} {dgx:>8.2f} {stxh:>8.2f} {format_percentage(pct):>10}")

print("\n" + "=" * 60)
print("ROCm w/ rocWMMA")
print("=" * 60)

print("\nPrompt Processing (pp) Tests:")
print(f"{'Test':<15} {'DGX':>8} {'STXH':>8} {'%':>10}")
print("-" * 45)
for test in ['pp2048', 'pp2048@d4096', 'pp2048@d8192', 'pp2048@d16384', 'pp2048@d32768']:
    dgx = dgx_data[test]
    stxh = stxh_rocm[test]
    pct = calculate_percentage(dgx, stxh)
    print(f"{test:<15} {dgx:>8.2f} {stxh:>8.2f} {format_percentage(pct):>10}")

print("\nToken Generation (tg) Tests:")
print(f"{'Test':<15} {'DGX':>8} {'STXH':>8} {'%':>10}")
print("-" * 45)
for test in ['tg32', 'tg32@d4096', 'tg32@d8192', 'tg32@d16384', 'tg32@d32768']:
    dgx = dgx_data[test]
    stxh = stxh_rocm[test]
    pct = calculate_percentage(dgx, stxh)
    print(f"{test:<15} {dgx:>8.2f} {stxh:>8.2f} {format_percentage(pct):>10}")
