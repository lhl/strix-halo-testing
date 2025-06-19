/* hip_v2_fix.h -- temporary workaround for ROCm ≥6.5 */
#ifdef __HIP_PLATFORM_AMD__
  #include <hipblas/hipblas.h>
  #ifndef HIPBLAS_V2
    #define HIPBLAS_V2 1   /* force new API paths in headers */
  #endif
  /* restore names llama.cpp expects */
  using hipblasDatatype_t  = hipDataType;
  using cudaDataType_t     = hipDataType;
  using cublasComputeType_t = hipblasComputeType_t;
  /* keep the old macro spellings alive */
  #ifndef CUBLAS_COMPUTE_16F
    #define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
    #define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
  #endif
#endif
