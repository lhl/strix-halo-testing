git clone https://github.com/ROCm/composable_kernel
cd composable_kernel

mkdir build
cd build

cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D GPU_ARCHS="gfx1100;gfx1151" \
    -D CMAKE_CXX_FLAGS="-U_GLIBCXX_ASSERTIONS" \
    ..
