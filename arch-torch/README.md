# Building Torch from Source for Strix Halo (gfx1151)

## Prerequisites
- Arch Linux
- ROCm/TheRock nightly installed at `/opt/rocm`
- Mamba/conda environment

## Setup

### 1. Create and activate environment
```bash
mamba create -n claude-torch python=3.12
mamba activate claude-torch
```

### 2. Run build scripts in order
The scripts are designed to run sequentially and build all dependencies:

```bash
# Install basic dependencies and tools
./00-setup-env.sh

# Build AOTriton (Flash Attention kernels)
./01-build-aotriton.sh

# Build ROCm WMMA library
./02-build-rocwmma.sh

# Build Composable Kernel library
./03-build-ck.sh

# Build PyTorch with ROCm and AOTriton support
./04-build-pytorch.sh
```

## Notes
- Scripts 00-03 can be run sequentially without modification
- AOTriton is built to a local install directory (`aotriton/build/install_dir`) 
- PyTorch script (04) is configured to use the locally built AOTriton
- All scripts are configured for Arch Linux package names
- Target GPU architectures: `gfx1100` and `gfx1151` (Strix Halo)

## Dependencies Built
- **AOTriton**: Flash Attention kernels for efficient attention computation
- **rocWMMA**: ROCm Warp Matrix Multiply-Accumulate library
- **Composable Kernel**: High-performance compute kernels for ROCm
- **PyTorch**: Deep learning framework with ROCm and AOTriton support

Additional docs: https://llm-tracker.info/_TOORG/Strix-Halo#pytorch-setup
