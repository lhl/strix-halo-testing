export LD_LIBRARY_PATH=/home/lhl/hipBLAS/build/release/hipblas-install/lib:/home/lhl/rocBLAS/build/release/rocblas-install/lib:$LD_LIBRARY_PATH
sudo ldconfig

# llama.cpp looks for so.4 not so.5... 
sudo ln -s /home/lhl/rocBLAS/build/release/rocblas-install/lib/librocblas.so.5 /home/lhl/rocBLAS/build/release/rocblas-install/lib/librocblas.so.4

# llama.cpp looks for so.2 not so.3... 
sudo ln -s /home/lhl/hipBLAS/build/release/hipblas-install/lib/libhipblas.so.3 /home/lhl/hipBLAS/build/release/hipblas-install/lib/libhipblas.so.2


git clone https://github.com/ggerganov/llama.cpp llama.cpp-hip
cd llama.cpp-hip

rm -rf build

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1100;gfx1151" -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -- -j 32

ldd build/bin/llama-bench | grep rocblas
