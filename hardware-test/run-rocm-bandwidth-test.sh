#!/bin/bash

# Clone only if directory doesn't exist
if [ ! -d "rocm_bandwidth_test" ]; then
    git clone https://github.com/ROCm/rocm_bandwidth_test
fi

cd rocm_bandwidth_test

# Build only if executable doesn't exist
if [ ! -f "build/rocm-bandwidth-test" ]; then
    CMAKE_MODULES=$(pwd)/cmake_modules
    mkdir -p build && cd build
    cmake -DCMAKE_MODULE_PATH=$CMAKE_MODULES -DCMAKE_PREFIX_PATH=$ROCM_PATH ..
    make
    cd ..
fi

# Set up ROCm library path using ROCM_PATH and run the test
export LD_LIBRARY_PATH=${ROCM_PATH:-/opt/rocm}/lib:$LD_LIBRARY_PATH
./build/rocm-bandwidth-test > ../rocm-bandwidth-test.out 2>&1
