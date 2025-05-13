# test_aotriton_pytorch.py
import torch
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

# Check environment variables
print("\nEnvironment variables:")
for var in ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', 'LD_LIBRARY_PATH', 'PYTORCH_ROCM_ARCH']:
    print(f"{var}: {os.environ.get(var, 'Not set')}")

# Test if AOTriton is loaded
try:
    import pyaotriton
    print("\npyaotriton imported successfully")
    
    # Check if PyTorch can see AOTriton
    if hasattr(torch.ops, 'aotriton'):
        print("torch.ops.aotriton is available")
    else:
        print("torch.ops.aotriton is NOT available")
    
    # Check registered operators
    if hasattr(torch.ops, 'aten'):
        print(f"Registered aten ops: {len(dir(torch.ops.aten))}")
except ImportError:
    print("Could not import pyaotriton")

# Test scaled_dot_product_attention with small tensors
if torch.cuda.is_available():
    print("\nTesting scaled_dot_product_attention...")
    device = 'cuda'
    batch, heads, seq_len, head_dim = 1, 1, 128, 64
    
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)
    
    try:
        result = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"Success! Result shape: {result.shape}")
    except Exception as e:
        print(f"Failed: {e}")
