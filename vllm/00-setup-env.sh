#!/usr/bin/env bash

set -e  # Exit on any error

# Configuration variables - fallback hierarchy: ENV_NAME -> CONDA_ENV -> "vllm"
ENV_NAME="${ENV_NAME:-${CONDA_ENV:-vllm}}"
PROJECT_NAME="${PROJECT_NAME:-vLLM}"
ENV_SETUP_ONLY="${ENV_SETUP_ONLY:-false}"

echo "=== $PROJECT_NAME Complete Environment Setup ==="

# Function to check if conda/mamba environment exists
env_exists() {
    conda env list | grep -q "^$ENV_NAME "
}

# Step 1: Create environment if it doesn't exist
if env_exists; then
    echo "✓ $ENV_NAME environment already exists"
else
    echo "Creating $ENV_NAME environment..."
    mamba create -n $ENV_NAME python=3.12 -y
    echo "✓ $ENV_NAME environment created"
fi

# Step 2: Initialize and activate environment
echo "Activating $ENV_NAME environment..."

# Initialize conda/mamba in this shell session
if [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    # Try to find conda installation
    CONDA_BASE=$(find /opt /usr/local $HOME -name "etc/profile.d/conda.sh" 2>/dev/null | head -1 | xargs dirname | xargs dirname | xargs dirname 2>/dev/null)
    if [ -n "$CONDA_BASE" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "✗ Error: Could not find conda installation!"
        echo "Please ensure conda/mamba is installed and accessible."
        exit 1
    fi
fi

# Initialize mamba if available
if [ -f "$HOME/mambaforge/etc/profile.d/mamba.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/mamba.sh"
fi

conda activate $ENV_NAME

# Ensure python and system commands are in PATH after activation
# export PATH="$CONDA_PREFIX/bin:/usr/bin:/usr/local/bin:$PATH"

# Step 3: Set up environment variables early (before PyTorch tests)
echo ""
echo "=== Setting Up Environment Variables ==="

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script location: $SCRIPT_DIR"

# Check if rocm-sdk is available (install ROCm first if needed)
if ! command -v rocm-sdk >/dev/null 2>&1; then
    echo "rocm-sdk not found, installing ROCm libraries first..."
    python -m pip install \
      --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/ \
      rocm[libraries,devel] -U --force-reinstall
fi

# Get ROCm paths from rocm-sdk
ROCM_PATH=$(rocm-sdk path --root 2>/dev/null)
ROCM_BIN_PATH=$(rocm-sdk path --bin 2>/dev/null)
ROCM_CMAKE_PATH=$(rocm-sdk path --cmake 2>/dev/null)

if [ -z "$ROCM_PATH" ]; then
    echo "✗ Failed to get ROCm root path from rocm-sdk!"
    exit 1
fi

echo "ROCm paths detected:"
echo "  Root: $ROCM_PATH"
echo "  Bin: $ROCM_BIN_PATH" 
echo "  CMake: $ROCM_CMAKE_PATH"

# AOTriton path (relative to script location)
AOTRITON_PATH="$SCRIPT_DIR/aotriton/build/install_dir"
echo "AOTriton path: $AOTRITON_PATH"

if [ ! -d "$AOTRITON_PATH" ]; then
    echo "⚠ AOTriton build directory not found at: $AOTRITON_PATH"
    echo "  This is normal if you haven't built AOTriton yet."
fi

# Remove any existing env vars first
conda env config vars unset ROCM_PATH HIP_PLATFORM HIP_PATH HIP_CLANG_PATH HIP_INCLUDE_PATH HIP_LIB_PATH HIP_DEVICE_LIB_PATH PATH LD_LIBRARY_PATH LIBRARY_PATH CPATH PKG_CONFIG_PATH AMD_SERIALIZE_KERNEL HIP_VISIBLE_DEVICES HIP_ARCH CUDA_HOME CUDA_PATH 2>/dev/null || true

# Set ROCm environment variables
conda env config vars set ROCM_PATH="$ROCM_PATH"
conda env config vars set HIP_PLATFORM="amd"
conda env config vars set HIP_PATH="$ROCM_PATH"
conda env config vars set HIP_INCLUDE_PATH="$ROCM_PATH/include"
conda env config vars set HIP_LIB_PATH="$ROCM_PATH/lib"
conda env config vars set CUDA_HOME="$CONDA_PREFIX"
conda env config vars set CUDA_PATH="$CONDA_PREFIX"

# Set HIP compiler paths using rocm-sdk detected paths
conda env config vars set HIP_CLANG_PATH="$ROCM_PATH/lib/llvm/bin"
conda env config vars set HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"

# Set PATH and library paths using rocm-sdk detected paths
conda env config vars set PATH="$CONDA_PREFIX/bin:$ROCM_BIN_PATH:$PATH"
conda env config vars set LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$AOTRITON_PATH/lib:\$LD_LIBRARY_PATH"
conda env config vars set LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:\$LIBRARY_PATH"

conda env config vars set CPATH="$ROCM_PATH/include:\$CPATH"
conda env config vars set PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig:\$PKG_CONFIG_PATH"

# Set debugging and device variables
conda env config vars set AMD_SERIALIZE_KERNEL="3"
conda env config vars set HIP_VISIBLE_DEVICES="0"
conda env config vars set HIP_ARCH="gfx1151"  # Strix Halo architecture

echo "✓ Environment variables configured"

# Show current config
echo ""
echo "=== Current Environment Configuration ==="
conda env config vars list

# If ENV_SETUP_ONLY is true, exit here
if [[ "$ENV_SETUP_ONLY" == "true" ]]; then
    echo ""
    echo "=== Environment Setup Complete (ENV_SETUP_ONLY=true) ==="
    echo "Environment variables have been set for $ENV_NAME environment."
    echo "To use the environment with new variables:"
    echo "  conda deactivate"
    echo "  conda activate $ENV_NAME"
    exit 0
fi

# Reactivate environment to load new variables
echo ""
echo "Reactivating environment to load new variables..."
conda deactivate
conda activate $ENV_NAME

# Step 4: Sanity check Python
echo ""
echo "=== Python Sanity Check ==="
echo "Current PATH: $PATH"
echo "Python location: $(command -v python)"
if command -v python >/dev/null 2>&1; then
    echo "Python version: $(python --version)"
else
    echo "Python version: python command not found"
fi
echo "Conda environment: $CONDA_DEFAULT_ENV"

if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "✗ Error: Not in $ENV_NAME environment!"
    exit 1
fi

# Step 5: Check existing PyTorch and optionally install TheRock ROCm and PyTorch
echo ""
echo "=== Checking Existing PyTorch Installation ==="

# Check if torch is already installed (don't exit on failure)
set +e  # Temporarily disable exit on error
TORCH_IMPORT_SUCCESS=false
if python -c "import torch" 2>/dev/null; then
    TORCH_IMPORT_SUCCESS=true
fi
set -e  # Re-enable exit on error

if [[ "$TORCH_IMPORT_SUCCESS" == "true" ]]; then
    echo "✓ PyTorch already installed:"
    python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  PyTorch ROCm version: {torch.version.hip}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f'  Device name: {torch.cuda.get_device_name()}')
"
    echo ""
    echo "Skipping TheRock PyTorch installation (already have custom build)"
    echo "If you want to install TheRock PyTorch, uninstall torch first:"
    echo "  pip uninstall torch -y"
    echo ""
    
    # Still install ROCm libraries if needed
    echo "=== Installing TheRock ROCm Libraries ==="
    python -m pip install \
      --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/ \
      rocm[libraries,devel] -U --force-reinstall
else
    echo "PyTorch not working or not installed, installing TheRock versions..."
    echo ""
    echo "=== Installing TheRock ROCm ==="
    python -m pip install \
      --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/ \
      rocm[libraries,devel] -U --force-reinstall

    echo ""
    echo "=== Installing TheRock PyTorch ==="
    python -m pip install \
      --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/ \
      torch -U --force-reinstall
fi

# if we want FA
# python -m pip install \
#  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx110X-dgpu/ \
#  torch torchaudio torchvision -U --force-reinstall

# Step 6: Test PyTorch installation with error handling
echo ""
echo "=== Testing PyTorch Installation ==="
set +e  # Temporarily disable exit on error
python -c "
try:
    import torch
    print(f'✓ PyTorch version: {torch.__version__}')
    print(f'✓ PyTorch ROCm version: {torch.version.hip}')
    print(f'✓ CUDA available: {torch.cuda.is_available()}')
    print(f'✓ Device count: {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        print(f'✓ Device name: {torch.cuda.get_device_name()}')
        print(f'✓ Device capability: {torch.cuda.get_device_capability()}')
        print('✓ PyTorch with ROCm working correctly!')
    else:
        print('⚠ CUDA not available - may need to reactivate environment')
except Exception as e:
    print(f'✗ PyTorch test failed: {e}')
    print('This may be expected if environment variables need to take effect')
"
set -e  # Re-enable exit on error

echo ""
echo "=== Setup Complete! ==="
echo "To use the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "The environment variables will be automatically set when you activate."
echo ""
echo "To test everything is working:"
echo "  conda deactivate && conda activate $ENV_NAME"
echo "  python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""

# Step 9: Test with new environment (requires reactivation)
echo ""
echo "=== Testing Current Setup ==="
echo "Note: Full test will work after reactivating the environment"

# Test basic imports
python -c "
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

# Step 10: Verify ROCm installation with environment variables
echo ""
echo "=== Verifying ROCm Installation ==="

echo "ROCm targets:"
rocm-sdk targets

echo ""
echo "Running ROCm tests:"
rocm-sdk test

echo ""
echo "=== Testing with Environment Variables ==="
echo "Reactivating environment to load new variables..."
conda deactivate
conda activate $ENV_NAME

echo ""
echo "Environment variables now active. Testing PyTorch with ROCm:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch ROCm version: {torch.version.hip}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Device capability: {torch.cuda.get_device_capability()}')
    print('✓ ROCm/CUDA working with environment variables!')
else:
    print('⚠ CUDA still not available - check environment configuration')
"

echo ""
echo "Environment setup complete!"
echo "The $ENV_NAME environment is now active with all ROCm variables configured."
