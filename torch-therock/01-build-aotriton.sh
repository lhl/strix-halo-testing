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
PINNED_COMMIT="1f9a37cdfbfce218fa0c07f5c0de40403019e168"
echo "Checking out AOTriton commit ${PINNED_COMMIT} (to match PyTorch)"
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

# Install Python site-package
# # Assuming your site-packages is at $SITE_PACKAGES
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp aotriton/build/install_dir/lib/pyaotriton.cpython-312-x86_64-linux-gnu.so $SITE_PACKAGES/
