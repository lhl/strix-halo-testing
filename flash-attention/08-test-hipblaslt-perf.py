# test_performance.py
import os
import torch
import torch.nn.functional as F
from triton.testing import do_bench

# Set environment
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1151'
os.environ['HIPBLASLT_TENSILE_LIBPATH'] = '/opt/rocm/lib/hipblaslt/library'
os.environ['TORCH_BLAS_PREFER_HIPBLASLT'] = '1'

device = 'cuda'
dtype = torch.float16

# Test GEMM performance
def test_gemm_performance():
    print("Testing GEMM performance...")
    m, n, k = 4096, 4096, 4096
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(10):
        c = torch.matmul(a, b)
    
    # Benchmark
    time_ms = do_bench(lambda: torch.matmul(a, b))
    flops = 2 * m * n * k
    tflops = flops / (time_ms * 1e9)
    
    print(f"GEMM {m}x{n}x{k}: {time_ms:.3f} ms, {tflops:.2f} TFLOPS")
    return tflops

# Test attention performance
def test_attention_performance():
    print("\nTesting Attention performance...")
    b, h, s, d = 8, 16, 2048, 64
    
    q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    k = torch.randn(b, h, s, d, device=device, dtype=dtype)
    v = torch.randn(b, h, s, d, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(10):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # Benchmark
    time_ms = do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    flops = 4 * b * h * s * s * d  # Approximate FLOPs for attention
    tflops = flops / (time_ms * 1e9)
    
    print(f"Attention {b}x{h}x{s}x{d}: {time_ms:.3f} ms, {tflops:.2f} TFLOPS")
    return tflops

# Check environment
print("Environment check:")
print(f"PYTORCH_ROCM_ARCH: {os.environ.get('PYTORCH_ROCM_ARCH', 'Not set')}")
print(f"HIPBLASLT_TENSILE_LIBPATH: {os.environ.get('HIPBLASLT_TENSILE_LIBPATH', 'Not set')}")
print(f"TORCH_BLAS_PREFER_HIPBLASLT: {os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT', 'Not set')}")

# Run tests
gemm_tflops = test_gemm_performance()
attn_tflops = test_attention_performance()

if gemm_tflops < 10:  # Should be much higher with proper hipBLASLt
    print("\nWARNING: GEMM performance is low. hipBLASLt may not be properly configured.")
    print("Check that:")
    print("1. The correct architecture kernels are in /opt/rocm/lib/hipblaslt/library")
    print("2. HIPBLASLT_TENSILE_LIBPATH is set correctly")
    print("3. Your GPU architecture matches the available kernels")
