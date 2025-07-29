#!/bin/bash

git clone https://github.com/ROCm/rocm_bandwidth_test
cd rocm_bandwidth_test
CMAKE_MODULES=pwd/cmake_modules
mkdir build && cd build
cmake -DCMAKE_MODULE_PATH=$CMAKE_MODULES -DCMAKE_PREFIX_PATH=$ROCM_PATH ..
./rocm-bandwidth-test > ../../rocm-bandwidth-test.out
