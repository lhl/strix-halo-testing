#!/usr/bin/env bash

set -e  # Exit on any error

# Configuration variables - fallback hierarchy: ENV_NAME -> CONDA_ENV -> "vllm"
ENV_NAME="${ENV_NAME:-${CONDA_ENV:-vllm}}"

echo "=== vLLM Build Script for gfx1151/Strix Halo ==="

# Step 1: Activate environment
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

# Verify we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "✗ Error: Not in $ENV_NAME environment!"
    exit 1
fi

echo "✓ $ENV_NAME environment activated"

# Step 2: We're already in the vLLM repository directory
echo "✓ Working in vLLM repository directory"

# Step 3: Install TheRock PyTorch (following the 00-setup-env.sh pattern)
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

# Step 4: Install build dependencies from README.md
echo ""
echo "=== Installing Build Dependencies ==="
pip install ninja cmake wheel pybind11

# Install AMD SMI (commented out - causes segfaults on gfx1151)
# pip install /opt/rocm/share/amd_smi || echo "⚠ AMD SMI install failed, continuing..."
echo "⚠ Skipping AMD SMI installation (causes segfaults on gfx1151)"

# Install other dependencies
pip install --upgrade numba scipy huggingface-hub[cli]
pip install "numpy<2"

# Step 5: Run use_existing_pytorch.py
echo ""
echo "=== Running use_existing_pytorch.py ==="
python use_existing_pytorch.py

# Step 6: Install ROCm build requirements
echo ""
echo "=== Installing ROCm Build Requirements ==="
pip install -r requirements/rocm-build.txt

# Step 7: Set PyTorch ROCm architecture
echo ""
echo "=== Setting PyTorch ROCm Architecture ==="
export PYTORCH_ROCM_ARCH="gfx1151"

# Step 8: Apply fixes for gfx1151 support
echo ""
echo "=== Applying gfx1151 Fixes ==="

# Fix 1: Add gfx1151 to CMakeLists.txt
echo "1. Adding gfx1151 to CMakeLists.txt HIP_SUPPORTED_ARCHS..."
if ! grep -q "gfx1151" CMakeLists.txt; then
    sed -i 's/set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1200;gfx1201")/set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1151;gfx1200;gfx1201")/' CMakeLists.txt
    echo "   ✓ Added gfx1151 to CMakeLists.txt"
else
    echo "   ✓ gfx1151 already present in CMakeLists.txt"
fi

# Fix 2: Remove torch dependency from pyproject.toml
echo "2. Removing torch dependency from pyproject.toml..."
if grep -q "torch == 2.8.0," pyproject.toml; then
    sed -i '/torch == 2.8.0,/d' pyproject.toml
    echo "   ✓ Removed torch dependency from pyproject.toml"
else
    echo "   ✓ torch dependency already removed from pyproject.toml"
fi

# Fix 3: Modify setup.py to handle missing torch in build isolation
echo "3. Patching setup.py to handle missing torch..."
if ! grep -q "TORCH_AVAILABLE = False" setup.py; then
    # Apply the torch import fix
    sed -i 's/import torch/try:\n    import torch\n    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME\n    TORCH_AVAILABLE = True\nexcept ImportError:\n    torch = None\n    CUDA_HOME = None\n    ROCM_HOME = None\n    TORCH_AVAILABLE = False/' setup.py
    
    # Fix other torch references
    sed -i 's/from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME/# Moved to try block above/' setup.py
    sed -i 's/torch.version.cuda is None/TORCH_AVAILABLE and torch.version.cuda is None/' setup.py
    sed -i 's/has_cuda = torch.version.cuda is not None/has_cuda = TORCH_AVAILABLE and torch.version.cuda is not None/' setup.py
    sed -i 's/torch.version.hip is not None/TORCH_AVAILABLE and torch.version.hip is not None/' setup.py
    sed -i 's/rocm_version = get_rocm_version() or torch.version.hip/rocm_version = get_rocm_version() or (torch.version.hip if TORCH_AVAILABLE else None)/' setup.py
    sed -i 's/cuda_major, cuda_minor = torch.version.cuda.split(".")/cuda_major, cuda_minor = torch.version.cuda.split(".") if TORCH_AVAILABLE else ("0", "0")/' setup.py
    
    echo "   ✓ Patched setup.py for torch handling"
else
    echo "   ✓ setup.py already patched for torch handling"
fi

# Fix 4: Fix ROCm platform detection to avoid amdsmi segfaults
echo "4. Patching ROCm platform detection to avoid amdsmi..."
if ! grep -q "Skip amdsmi check due to segfault issues" vllm/platforms/__init__.py; then
    # Replace the amdsmi detection with torch-based detection
    sed -i '/def rocm_platform_plugin/,/return "vllm.platforms.rocm.RocmPlatform" if is_rocm else None/{
        s/is_rocm = False/is_rocm = False/
        s/logger.debug("Checking if ROCm platform is available.")/logger.debug("Checking if ROCm platform is available.")\n    \n    # Skip amdsmi check due to segfault issues - default to ROCm for AMD systems/
        s/try:\n        import amdsmi/try:\n        import torch/
        s/amdsmi.amdsmi_init()/# amdsmi disabled - using torch detection/
        s/try:\n            if len(amdsmi.amdsmi_get_processor_handles()) > 0:/if hasattr(torch, '\''version'\'') and hasattr(torch.version, '\''hip'\'') and torch.version.hip is not None:/
        s/is_rocm = True\n                logger.debug("Confirmed ROCm platform is available.")/is_rocm = True\n            logger.debug("ROCm platform detected via torch.version.hip")/
        s/else:\n                logger.debug("ROCm platform is not available because"\n                             " no GPU is found.")/else:\n            # Fallback: assume ROCm if we'\''re not CUDA and not other platforms\n            logger.debug("Defaulting to ROCm platform (amdsmi disabled due to segfault)")\n            is_rocm = True/
        s/finally:\n            amdsmi.amdsmi_shut_down()//
        s/logger.debug("ROCm platform is not available because: %s", str(e))/logger.debug("ROCm platform check failed: %s", str(e))\n        # Still default to ROCm as fallback\n        is_rocm = True/
    }' vllm/platforms/__init__.py
    echo "   ✓ Patched ROCm platform detection"
else
    echo "   ✓ ROCm platform detection already patched"
fi

echo "✓ All gfx1151 fixes applied"

# Step 9: Build vLLM with ROCm target device
echo ""
echo "=== Building vLLM ==="
echo "Using pip install . -e --no-build-isolation (to use our torch correctly)"

VLLM_TARGET_DEVICE=rocm pip install . -e --no-build-isolation

# Note: python setup.py develop doesn't work, so we use pip install instead

# Final numpy fix
pip install "numpy<2"

echo ""
echo "=== vLLM Build Complete ==="
echo ""
echo "To test vLLM installation:"
echo "  conda activate $ENV_NAME"
echo "  python -c \"import vllm; print('vLLM version:', vllm.__version__)\""
echo ""
echo "To download ShareGPT dataset and run benchmark:"
echo "  wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
echo "  python benchmark_serving.py --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 64 --max-concurrency 1 --model unsloth/Llama-3.2-1B-Instruct"