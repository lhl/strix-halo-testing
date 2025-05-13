# test_attention_backend.py
import torch
import os
import torch.nn.functional as F
from triton.testing import do_bench

# Set all necessary environment variables
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1151'
os.environ['HIPBLASLT_TENSILE_LIBPATH'] = '/opt/rocm/lib/hipblaslt/library'
os.environ['TORCH_BLAS_PREFER_HIPBLASLT'] = '1'
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'

print("=== Environment ===")
for key in ['PYTORCH_ROCM_ARCH', 'HIPBLASLT_TENSILE_LIBPATH', 'TORCH_BLAS_PREFER_HIPBLASLT', 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL']:
    print(f"{key}: {os.environ.get(key, 'Not set')}")

print("\n=== Backend Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

# Check if AOTriton is available
print("\n=== AOTriton Check ===")
try:
    import pyaotriton
    print("pyaotriton imported successfully")
    if hasattr(torch.ops, 'aotriton'):
        print("torch.ops.aotriton is available")
        print(f"AOTriton ops: {dir(torch.ops.aotriton)}")
    else:
        print("torch.ops.aotriton is NOT available")
except ImportError as e:
    print(f"Could not import pyaotriton: {e}")

# Check available backends for SDPA
print("\n=== SDPA Backends ===")
if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
    print(f"Flash SDPA enabled: {torch.backends.cuda.flash_sdp_enabled()}")
if hasattr(torch.backends.cuda, 'mem_efficient_sdp_enabled'):
    print(f"Memory efficient SDPA enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
if hasattr(torch.backends.cuda, 'math_sdp_enabled'):
    print(f"Math SDPA enabled: {torch.backends.cuda.math_sdp_enabled()}")

# Test different attention configurations
print("\n=== Testing Attention Variants ===")

def test_attention_variant(name, func, b=8, h=16, s=2048, d=64):
    device = 'cuda'
    dtype = torch.float16
    
    q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    k = torch.randn(b, h, s, d, device=device, dtype=dtype)
    v = torch.randn(b, h, s, d, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(10):
        _ = func(q, k, v)
    
    # Benchmark
    time_ms = do_bench(lambda: func(q, k, v))
    flops = 4 * b * h * s * s * d  # Approximate FLOPs
    tflops = flops / (time_ms * 1e9)
    
    print(f"{name}: {time_ms:.3f} ms, {tflops:.2f} TFLOPS")
    return tflops

# Test different attention implementations
print("\n1. Standard SDPA (no causal):")
test_attention_variant(
    "Standard SDPA",
    lambda q, k, v: F.scaled_dot_product_attention(q, k, v)
)

print("\n2. Causal SDPA:")
test_attention_variant(
    "Causal SDPA",
    lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
)

print("\n3. SDPA with attn_mask:")
def sdpa_with_mask(q, k, v):
    mask = torch.tril(torch.ones(q.size(-2), k.size(-2), device=q.device))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

test_attention_variant(
    "SDPA with mask",
    sdpa_with_mask
)

print("\n4. Force Flash Attention backend:")
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    test_attention_variant(
        "Flash Attention",
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    )

print("\n5. Force Math backend:")
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    test_attention_variant(
        "Math backend",
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    )

# Check which backend is actually being used
print("\n=== Backend Selection Debug ===")
def check_backend_selection():
    import torch._dynamo
    
    # Create a simple model to trace
    class AttentionModel(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    model = AttentionModel()
    q = torch.randn(1, 1, 128, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 1, 128, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 1, 128, 64, device='cuda', dtype=torch.float16)
    
    # Try to trace and see what backend is selected
    try:
        traced = torch.jit.trace(model, (q, k, v))
        print("JIT trace successful")
    except Exception as e:
        print(f"JIT trace failed: {e}")
    
    # Try with torch.compile
    try:
        compiled_model = torch.compile(model)
        result = compiled_model(q, k, v)
        print("torch.compile successful")
    except Exception as e:
        print(f"torch.compile failed: {e}")

check_backend_selection()

# Additional debugging
print("\n=== Additional Debug Info ===")
if torch.cuda.is_available():
    print(f"CUDA arch list: {torch.cuda.get_arch_list()}")
    print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    
    # Check if we can manually call Flash Attention
    try:
        from torch.nn.functional import _scaled_dot_product_flash_attention_forward
        print("Flash Attention forward function is available")
    except ImportError:
        print("Flash Attention forward function is NOT available")
