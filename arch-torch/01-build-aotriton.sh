#!/bin/bash 

if [ ! -d "aotriton" ]; then
    git clone https://github.com/ROCm/aotriton
fi
cd aotriton
git submodule sync && git submodule update --init --recursive --force

if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi
mkdir build && cd build

cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_GPU_BUILD_TIMEOUT=0 -DAOTRITON_TARGET_ARCH="gfx1100;gfx1151" -G Ninja

ninja install
