#!/usr/bin/env bash

# Activate the conda environment if not already active
if [[ "$CONDA_DEFAULT_ENV" != "therock" ]]; then
    echo "Activating therock environment..."
    conda activate therock
fi

# Get the Python site-packages directory where PyTorch is installed
PYTORCH_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")

if [ -z "$PYTORCH_PATH" ]; then
    echo "Error: Could not find PyTorch installation path"
    exit 1
fi

echo "Found PyTorch at: $PYTORCH_PATH"

if [ ! -d "aotriton" ]; then
    git clone https://github.com/ROCm/aotriton
fi
cd aotriton
# Checkout the PyTorch-pinned commit to avoid API mismatches
# Prefer the pin from PyTorch's aotriton.cmake if available
DEFAULT_PIN="1f9a37cdfbfce218fa0c07f5c0de40403019e168"
AOTRITON_CMAKE_PIN_FILE="../TheRock/external-builds/pytorch/pytorch/cmake/External/aotriton.cmake"
if [ -f "$AOTRITON_CMAKE_PIN_FILE" ]; then
    PINNED_COMMIT=$(awk -F '"' '/set\(__AOTRITON_CI_COMMIT/{print $2; exit}' "$AOTRITON_CMAKE_PIN_FILE")
    if [[ -z "$PINNED_COMMIT" ]]; then
        PINNED_COMMIT="$DEFAULT_PIN"
    fi
else
    PINNED_COMMIT="$DEFAULT_PIN"
fi
echo "AOTriton pin: ${PINNED_COMMIT} (source: ${AOTRITON_CMAKE_PIN_FILE:-default})"
git fetch --all --tags --prune
git checkout --detach "$PINNED_COMMIT"
git submodule sync && git submodule update --init --recursive --force

if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi
mkdir build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH="$PYTORCH_PATH" \
  -DCMAKE_INSTALL_PREFIX=./install_dir \
  -DCMAKE_BUILD_TYPE=Release \
  -DAOTRITON_GPU_BUILD_TIMEOUT=0 \
  -DAOTRITON_TARGET_ARCH="gfx1151" \
  -G Ninja

ninja install

# Install Python site-package and shared libs (folded in from install-aotriton.sh)
# Resolve install dir from current build tree to avoid relative path issues
INSTALL_DIR="$(pwd)/install_dir"
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
TORCH_LIB_DIR="$SITE_PACKAGES/torch/lib"

# Ensure torch lib dir exists
mkdir -p "$TORCH_LIB_DIR"

# Find built pyaotriton .so (handles different Python ABI suffixes)
PYAOTRITON_SO=$(ls "$INSTALL_DIR/lib"/pyaotriton*.so 2>/dev/null | head -n1 || true)
if [ -z "$PYAOTRITON_SO" ]; then
    echo "Error: pyaotriton shared object not found under $INSTALL_DIR/lib" >&2
    exit 1
fi

# Copy pyaotriton next to site-packages root (importable) and next to torch/lib (runtime deps colocated)
cp "$PYAOTRITON_SO" "$SITE_PACKAGES/"
cp "$PYAOTRITON_SO" "$TORCH_LIB_DIR/"

# Copy libaotriton_v2 and set soname symlink in torch/lib
LIBAOTRITON_VER_PATH=$(ls "$INSTALL_DIR/lib"/libaotriton_v2.so.* 2>/dev/null | head -n1 || true)
if [ -n "$LIBAOTRITON_VER_PATH" ]; then
    cp "$LIBAOTRITON_VER_PATH" "$TORCH_LIB_DIR/"
    LIBAOTRITON_VER_BASENAME=$(basename "$LIBAOTRITON_VER_PATH")
    (cd "$TORCH_LIB_DIR" && ln -sf "$LIBAOTRITON_VER_BASENAME" libaotriton_v2.so)
else
    # Fallback: if unversioned lib exists in install tree, copy it directly
    if [ -f "$INSTALL_DIR/lib/libaotriton_v2.so" ]; then
        cp "$INSTALL_DIR/lib/libaotriton_v2.so" "$TORCH_LIB_DIR/"
    else
        echo "Warning: libaotriton_v2.so not found in $INSTALL_DIR/lib; PyTorch runtime may not load AOTriton." >&2
    fi
fi

# Copy compiled kernel images if present
if [ -d "$INSTALL_DIR/lib/aotriton.images" ]; then
    mkdir -p "$TORCH_LIB_DIR/aotriton.images"
    cp -r "$INSTALL_DIR/lib/aotriton.images/"* "$TORCH_LIB_DIR/aotriton.images/" 2>/dev/null || true
fi

echo "You should be able to run `python 10-test_aotriton_direct.py` to test. If you get errors, give `install-aotriton.sh` a try for helping to install." 
