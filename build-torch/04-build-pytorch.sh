# we need rocm-cmake
git clone https://github.com/ROCm/rocm-cmake rocm-cmake                                                                                                                                                                                                          │
cmake -S rocm-cmake -B rocm-cmake/build -DCMAKE_INSTALL_PREFIX=/opt/rocm                                                                                                                                                                                         │
sudo cmake --install rocm-cmake/build

# also rocm-core 
git clone https://github.com/ROCm/rocm-core.git
mkdir -p rocm-core/build
cmake -S rocm-core -B rocm-core/build -DCMAKE_INSTALL_PREFIX=/opt/rocm -DROCM_VERSION=6.5.0
cmake --build rocm-core/build -j$(nproc)
sudo cmake --install rocm-core/build

# pytorch time
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule sync && git submodule update --init --recursive --force

# Enable ROCm (HIP) build and disable CUDA
export USE_ROCM=1
export USE_CUDA=0

# DISABLE KINETO
export USE_KINETO=OFF

# still needed for ROCM_ROCTX_LIB
sudo dnf install roctracer-devel
ln -s /opt/rocm/lib/librocprofiler-sdk-roctx.so /opt/rocm/lib/libroctx64.so

# Will complain about tracing which we're not building...
export BUILD_TEST=OFF

# Needed
sudo dnf install libdrm-devel

# for benchmark.h? - or export BUILD_TEST=OFF
sudo dnf install google-benchmark-devel

# Enable AOTriton integration (FlashAttention kernels) - flag changed w/ 2.8?
export USE_AOTRITON=1
export BUILD_AOTRITON=0

# Specify target GPU architectures for ROCm (limit to gfx1151 for Strix Halo)
export PYTORCH_ROCM_ARCH="gfx1151"

# Point to pre-installed AOTriton (adjust the path to your AOTriton install dir)
export AOTRITON_INSTALLED_PREFIX="/opt/rocm"

# Add ROCm and custom library paths to CMake search path
export CMAKE_PREFIX_PATH="/opt/rocm:${CMAKE_PREFIX_PATH}"

# Ensure ROCm libs (and any custom build libs) are in the runtime library path
export LD_LIBRARY_PATH="/opt/rocm/lib:${AOTRITON_INSTALLED_PREFIX}/lib:${LD_LIBRARY_PATH}"

export CXXFLAGS="$CXXFLAGS -Wno-unused-function -Wno-error=unused-variable -Wno-error=unused-function -Wno-error=deprecated-declarations -Wno-error=switch -Wno-error=unused-local-typedefs  -Wno-error=calloc-transposed-args -Wno-array-bounds -Wno-error=array-bounds -Wno-dangling-pointer -Wno-stringop-overread"

# gcc15 doesn't include default lib headers anymore that tensorpipe and gloo depend on
export CXXFLAGS="-include cstdint ${CXXFLAGS}"

# also new in gcc15
export CXXFLAGS="$CXXFLAGS -Wno-free-nonheap-object -Wno-error=free-nonheap-object"

# more gcc15
export CXXFLAGS="-D_GLIBCXX_ASSERTIONS=0 $CXXFLAGS"

# hipside needs as well
export HIPCC_COMPILE_FLAGS_APPEND="-include cstdint -D_GLIBCXX_ASSERTIONS=0"


# export CXXFLAGS="$CXXFLAGS -Wno-error"
# export CCFLAGS="$CFLAGS -Wno-error"
# export HIPCC_FLAGS="$HIPCC_FLAGS -Wno-error"   # for hipcc-compiled kernels


We need to add
defined(__gfx1151__) || 
to
third_party/composable_kernel/include/ck/ck.hpp

# Before we start compiling we need to hipify:
python tools/amd_build/build_amd.py

# see below for rocm-cmake

# see below for rocm-core

pip install -r requirements.txt

# If using CI, modify for STATIC benchmarks OFF
time .ci/pytorch/build.sh

# or just try to directly run:
# cmake3 --build . --target install --config Release

# To get things working/installed properly...
python setup.py develop && python -c "import torch"


# Does this work?
python -c 'import torch,os; print(torch.version.hip, torch.cuda.get_device_name(0))'

# python - <<'PY'
import torch
print("HIP runtime:", torch.version.hip)
print("Device:", torch.cuda.get_device_name(0))
PY
HIP runtime: 6.4.43480-9f04e2822
Device: AMD Radeon Graphics
