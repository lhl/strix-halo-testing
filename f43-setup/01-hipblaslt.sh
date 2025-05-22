# We need a version that has gfx1151 support

sudo dnf install msgpack-devel -y
sudo dnf install roctracer-devel -y
sudo dnf install lld-devel libffi-devel -y
sudo dnf install rocm-cmake -y
sudo dnf install blas-devel -y
sudo dnf install ccache -y

# client
sudo dnf install gfortran -y
sudo dnf install blas-static lapack-static -y
sudo dnf install rocm-smi-devel -y
sudo dnf install gtest-devel -y
sudo dnf install rocm-omp-devel -y

git clone https://github.com/ROCm/hipBLASLt
cd hipBLASLt
python3 -m pip install -r tensilelite/requirements.txt

# export CXX variables doesn't help...
sudo mkdir -p /opt/rocm/bin
sudo ln -s /usr/lib64/rocm/llvm/bin/clang++ /opt/rocm/bin/amdclang++
sudo ln -s /usr/lib64/rocm/llvm/bin/clang /opt/rocm/bin/amdclang

# rocm-llvm-devel and rocm-lld don't have cmake, lld-devel is v20 (rocm is v19) 
sudo mkdir -p /usr/lib64/rocm/llvm/lib/cmake/lld
cat <<'EOF' | sudo tee /usr/lib64/rocm/llvm/lib/cmake/lld/LLDConfig.cmake
# minimal stub for ROCm on Fedora
include("${CMAKE_CURRENT_LIST_DIR}/../llvm/LLVMConfig.cmake")
set(LLD_FOUND TRUE)
EOF


export HIP_PLATFORM=amd
export HIPBLASLT_ENABLE_MARKER=0

# may still have to manually edit this in install.sh... 
export skip_rocroller=true
export use_rocroller=false


# Setup
./install.sh -id -a "gfx1100;gfx1151"
# ./install.sh -idc -a "gfx1100;gfx1151"


# Install to /usr/local
sudo cmake --install build/release --prefix /usr/local
export LD_LIBRARY_PATH=/usr/local/lib

# You might want to move it to /usr/lib64 which is where the system version is

# Test if it's working
./hipblaslt-test
