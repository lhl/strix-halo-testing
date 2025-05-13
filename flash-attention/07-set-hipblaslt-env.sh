# Set these before running Python
export PYTORCH_ROCM_ARCH="gfx1151"
export ROCM_PATH="/opt/rocm"
export HIP_PATH="/opt/rocm"
export ROCBLAS_LAYER=gemm
export HIPBLASLT_TENSILE_LIBPATH="/opt/rocm/lib/hipblaslt/library"
export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
