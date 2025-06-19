git clone https://github.com/ROCm/rocWMMA

# Change FP8 check from FAIL to STATUS

rm -rf build; mkdir build; CC=/opt/rocm/llvm/bin/amdclang CXX=/opt/rocm/llvm/bin/amdclang++ cmake -B build  . -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/rocm -DROCWMMA_BUILD_TESTS=OFF -DROCWMMA_BUILD_SAMPLES=OFF GPU_TARGETS="gfx1100;gfx1151"

cmake --build build -j$(nproc)
sudo cmake --install build
