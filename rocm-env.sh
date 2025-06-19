# ---- ROCm nightly from /opt/rocm ---------------------------------
export ROCM_PATH=/opt/rocm           # canonical root
export HIP_PLATFORM=amd
export HIP_PATH=$ROCM_PATH           # some tools still look for it
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export HIP_INCLUDE_PATH=$ROCM_PATH/include
export HIP_LIB_PATH=$ROCM_PATH/lib
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode   # device bitcode libs

# search paths
export PATH=$ROCM_PATH/bin:$HIP_CLANG_PATH:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LIBRARY_PATH
export CPATH=$HIP_INCLUDE_PATH:$CPATH           # for clang/gcc
export PKG_CONFIG_PATH=$ROCM_PATH/lib/pkgconfig:$PKG_CONFIG_PATH

