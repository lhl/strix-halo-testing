# Create a modified benchmark script
# test_attention_small.py
import torch
import torch.nn.functional as F
from triton.testing import do_bench

def test_small_attention():
    device = 'cuda'
    batch, heads, seq_len, head_dim = 1, 1, 128, 64  # Much smaller
    
    print(f"Testing with sizes: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")
    
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    # Test basic attention
    try:
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"Basic attention success! Result shape: {result.shape}")
    except Exception as e:
        print(f"Basic attention failed: {e}")
    
    # Test with AOTriton environment variable
    import os
    os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
    
    try:
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"AOTriton attention success! Result shape: {result.shape}")
    except Exception as e:
        print(f"AOTriton attention failed: {e}")

if __name__ == "__main__":
    test_small_attention()
