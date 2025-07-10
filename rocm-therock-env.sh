# ---- ROCm nightly from /home/lhl/therock/rocm-7.0 ----
export ROCM_PATH=/home/lhl/therock/rocm-7.0
export HIP_PLATFORM=amd
export HIP_PATH=$ROCM_PATH
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export HIP_INCLUDE_PATH=$ROCM_PATH/include
export HIP_LIB_PATH=$ROCM_PATH/lib
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode

# search paths -- prepend
export PATH="$ROCM_PATH/bin:$HIP_CLANG_PATH:$PATH"
export LD_LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:${LIBRARY_PATH:-}"
export CPATH="$HIP_INCLUDE_PATH:${CPATH:-}"
export PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
