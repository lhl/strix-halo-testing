git clone https://github.com/ROCm/composable_kernel
cd composable_kernel

rm -rf build
mkdir build
cd build

CC=/opt/rocm/llvm/bin/amdclang CXX=/opt/rocm/llvm/bin/amdclang++ cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
    -D CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -D GPU_TARGETS="gfx1100;gfx1151" \
    -D CMAKE_CXX_FLAGS="-U_GLIBCXX_ASSERTIONS" \
    ..

ninja -j$(nproc)
sudo ninja install
