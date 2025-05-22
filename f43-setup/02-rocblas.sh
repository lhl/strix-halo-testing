git clone https://github.com/ROCm/rocBLAS
cd rocBLAS
HIP_PLATFORM=amd ./install.sh -a 'gfx1100;gfx1151'
export LD_LIBRARY_PATH=/home/lhl/rocBLAS/build/release/rocblas-install/lib:$LD_LIBRARY_PATH
