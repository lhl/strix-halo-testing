#!/usr/bin/env bash

set -e  # Exit on any error

# Get environment name from argument or default to "therock"
ENV_NAME="${1:-therock}"

echo "=== TheRock Complete Environment Setup ==="
echo "Target environment: $ENV_NAME"

# Function to check if conda/mamba environment exists
env_exists() {
    conda env list | grep -q "^$ENV_NAME "
}

# Step 1: Create environment if it doesn't exist
if env_exists; then
    echo "✓ $ENV_NAME environment already exists"
else
    echo "Creating $ENV_NAME environment..."
    mamba create -n "$ENV_NAME" python=3.12 -y
    echo "✓ $ENV_NAME environment created"
fi

# Step 2: Initialize conda for configuration
echo "Initializing conda for configuration..."

# Detect conda executable path
CONDA_EXE="${CONDA_EXE:-$(which conda 2>/dev/null)}"
if [ -z "$CONDA_EXE" ]; then
    echo "✗ Error: Could not find conda executable!"
    echo "Please ensure conda/mamba is installed and accessible."
    exit 1
fi

# Get conda base from CONDA_EXE
CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
echo "Detected conda installation: $CONDA_BASE"

# Set CONDA_PREFIX for target environment
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
if [ ! -d "$ENV_PREFIX" ]; then
    echo "✗ Error: $ENV_NAME environment not found at $ENV_PREFIX"
    exit 1
fi

echo "Working with $ENV_NAME environment at: $ENV_PREFIX"

# Step 3: Install packages in environment
echo ""
echo "=== Installing TheRock ROCm ==="
"$ENV_PREFIX/bin/python" -m pip install \
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/ \
  rocm[libraries,devel] -U --force-reinstall

echo ""
echo "=== Installing TheRock PyTorch ==="
"$ENV_PREFIX/bin/python" -m pip install \
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/ \
  torch -U --force-reinstall

# if we want FA
# python -m pip install \
#  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx110X-dgpu/ \
#  torch torchaudio torchvision -U --force-reinstall

# Step 4: Test PyTorch installation
echo ""
echo "=== Testing PyTorch Installation ==="
"$ENV_PREFIX/bin/python" -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch ROCm version: {torch.version.hip}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Device capability: {torch.cuda.get_device_capability()}')
else:
    print('⚠ CUDA not available - may need environment variables')
"

# Step 6: Detect ROCm paths
echo ""
echo "=== Detecting ROCm Installation ==="

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script location: $SCRIPT_DIR"

# Get PyTorch ROCm version for matching
PYTORCH_ROCM=$("$ENV_PREFIX/bin/python" -c "import torch; print(torch.version.hip)" 2>/dev/null || echo "unknown")
echo "PyTorch was built with ROCm: $PYTORCH_ROCM"

# Check available ROCm installations (prioritize conda-installed nightly over system)
# Use rocm-sdk to determine ROCm paths
echo "Detecting ROCm installation via rocm-sdk..."

# Check if rocm-sdk is available in the environment
if ! "$ENV_PREFIX/bin/python" -c "import rocm_sdk" 2>/dev/null; then
    echo "✗ rocm-sdk module not found in $ENV_NAME environment!"
    echo "ROCm SDK should be installed via pip during the ROCm installation step above."
    exit 1
fi

# Get ROCm paths from rocm-sdk using the environment's python
ROCM_PATH=$("$ENV_PREFIX/bin/python" -m rocm_sdk path --root 2>/dev/null)
ROCM_BIN_PATH=$("$ENV_PREFIX/bin/python" -m rocm_sdk path --bin 2>/dev/null)
ROCM_CMAKE_PATH=$("$ENV_PREFIX/bin/python" -m rocm_sdk path --cmake 2>/dev/null)

if [ -z "$ROCM_PATH" ]; then
    echo "✗ Failed to get ROCm root path from rocm-sdk!"
    exit 1
fi

echo "ROCm paths detected:"
echo "  Root: $ROCM_PATH"
echo "  Bin: $ROCM_BIN_PATH" 
echo "  CMake: $ROCM_CMAKE_PATH"



# Step 7: Set conda environment variables
echo ""
echo "=== Setting Conda Environment Variables ==="

# AOTriton path (relative to script location)
AOTRITON_PATH="$SCRIPT_DIR/aotriton/build/install_dir"
echo "AOTriton path: $AOTRITON_PATH"

if [ ! -d "$AOTRITON_PATH" ]; then
    echo "⚠ AOTriton build directory not found at: $AOTRITON_PATH"
    echo "  This is normal if you haven't built AOTriton yet."
    echo "  Run the AOTriton build script first, then re-run this setup."
fi

# Remove any existing env vars first (for the target environment)
echo "Clearing existing environment variables..."
conda env config vars unset \
    ROCM_PATH HIP_PLATFORM HIP_PATH HIP_CLANG_PATH \
    HIP_INCLUDE_PATH HIP_LIB_PATH HIP_DEVICE_LIB_PATH \
    LD_LIBRARY_PATH LIBRARY_PATH CPATH PKG_CONFIG_PATH \
    AMD_SERIALIZE_KERNEL HIP_VISIBLE_DEVICES HIP_ARCH \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL PYTORCH_ROCM_ARCH \
    TORCH_ROCM_AOTRITON_PREFER_DEFAULT \
    -n "$ENV_NAME" 2>/dev/null || true

# Set ROCm environment variables
echo "Setting ROCm environment variables..."
conda env config vars set \
    ROCM_PATH="$ROCM_PATH" \
    HIP_PLATFORM="amd" \
    HIP_PATH="$ROCM_PATH" \
    HIP_INCLUDE_PATH="$ROCM_PATH/include" \
    HIP_LIB_PATH="$ROCM_PATH/lib" \
    HIP_CLANG_PATH="$ROCM_PATH/lib/llvm/bin" \
    HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode" \
    -n "$ENV_NAME"

# Set library paths (note: we don't set PATH here as it causes conflicts)
echo "Setting library paths..."
conda env config vars set \
    LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$AOTRITON_PATH/lib" \
    LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64" \
    CPATH="$ROCM_PATH/include" \
    PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig" \
    -n "$ENV_NAME"

# Set debugging and device variables
echo "Setting device and runtime variables..."
conda env config vars set \
    AMD_SERIALIZE_KERNEL="0" \
    HIP_VISIBLE_DEVICES="0" \
    HIP_ARCH="gfx1151" \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1" \
    PYTORCH_ROCM_ARCH="gfx1151" \
    TORCH_ROCM_AOTRITON_PREFER_DEFAULT="1" \
    -n "$ENV_NAME"

echo "✓ Environment variables configured"

# Create activation script to add ROCm bin to PATH
# This is needed because conda env vars don't handle PATH well
echo "Creating activation script for PATH..."
ACTIVATE_DIR="$ENV_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat > "$ACTIVATE_DIR/rocm_path.sh" << EOF
#!/bin/bash
# Add ROCm bin to PATH
export PATH="$ROCM_BIN_PATH:\$PATH"
EOF
chmod +x "$ACTIVATE_DIR/rocm_path.sh"
echo "✓ Activation script created at $ACTIVATE_DIR/rocm_path.sh"

# Helper: print a convenience one-liner and suggest helper script
echo ""
echo "You can quickly run the backend check with:" 
echo "  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTORCH_ROCM_ARCH=gfx1151 TORCH_ROCM_AOTRITON_PREFER_DEFAULT=1 HIP_VISIBLE_DEVICES=0 python backend-check.py"
echo "Or use the helper script: ./run-backend-check.sh"

# Step 8: Show current config
echo ""
echo "=== Current Environment Configuration ==="
conda env config vars list -n "$ENV_NAME"

echo ""
echo "=== Setup Complete! ==="
echo "To use the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "The environment variables will be automatically set when you activate."
echo ""
echo "To test everything is working:"
echo "  conda activate $ENV_NAME"
echo "  python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
echo ""
echo "Note: You may need to deactivate and reactivate the environment for"
echo "the new environment variables to take effect."

# Step 9: Test with new environment (requires reactivation)
echo ""
echo "=== Testing Current Setup ==="
echo "Note: Full test will work after reactivating the environment"

# Test basic imports
"$ENV_PREFIX/bin/python" -c "
try:
    import torch
    print('✓ PyTorch imports successfully')
    print(f'  Version: {torch.__version__}')
    print(f'  ROCm version: {torch.version.hip}')
except Exception as e:
    print(f'✗ PyTorch import failed: {e}')

try:
    import pyaotriton
    print('✓ AOTriton imports successfully')
except Exception as e:
    print(f'? AOTriton import status: {e}')
    print('  (This is expected if AOTriton is not yet installed)')
"

# Step 10: Verify ROCm installation
echo ""
echo "=== Verifying ROCm Installation ==="

echo "ROCm targets:"
"$ENV_PREFIX/bin/python" -m rocm_sdk targets

echo ""
echo "Running ROCm tests:"
"$ENV_PREFIX/bin/python" -m rocm_sdk test

echo ""
echo "=== Note About Environment Variables ==="
echo "Environment variables have been configured for the '$ENV_NAME' conda environment."
echo "They will be automatically loaded when you activate the environment in a new shell:"
echo "  conda deactivate"
echo "  conda activate $ENV_NAME"
echo ""
echo "In this script session, the environment variables are NOT yet active because"
echo "conda env vars are only loaded when activating in an interactive shell."
echo ""
echo "You can verify the setup works by running in a new shell:"
echo "  conda activate $ENV_NAME"
echo "  python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
echo ""
echo "Environment setup complete!"
echo "The $ENV_NAME environment variables have been configured."
