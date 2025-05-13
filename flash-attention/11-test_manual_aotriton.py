# test_manual_aotriton.py
import torch
import pyaotriton
import triton
from triton.testing import do_bench

print("=== Manual AOTriton Test ===")

# Check available functions in AOTriton
print("AOTriton functions:")
if hasattr(pyaotriton, 'v2'):
    print(f"v2 module: {dir(pyaotriton.v2)}")
    if hasattr(pyaotriton.v2, 'flash'):
        print(f"flash module: {dir(pyaotriton.v2.flash)}")
        try:
            from pyaotriton.v2.flash import attn_fwd, attn_bwd
            print("Successfully imported attention functions!")
            
            # Test with actual tensors
            device = 'cuda'
            dtype = torch.float16
            batch_size = 2
            num_heads = 8
            seq_len = 1024
            head_dim = 64
            
            # Create tensors
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                          device=device, dtype=dtype)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                          device=device, dtype=dtype)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                          device=device, dtype=dtype)
            
            # Try to call AOTriton attention
            print("\nTesting AOTriton attention...")
            try:
                # Check function signature
                import inspect
                print(f"attn_fwd signature: {inspect.signature(attn_fwd)}")
                
                # AOTriton might expect different tensor shapes or additional parameters
                # Let's try different approaches
                
                # Approach 1: Direct call
                try:
                    output = attn_fwd(q, k, v, None, 0.125, True, None)
                    print("Direct call successful!")
                except Exception as e:
                    print(f"Direct call failed: {e}")
                
                # Approach 2: Check if there's a wrapper or different API
                if hasattr(pyaotriton.v2.flash, 'attention'):
                    print("Found attention function in flash module")
                
            except Exception as e:
                print(f"AOTriton call failed: {e}")
                
        except ImportError as e:
            print(f"Could not import flash functions: {e}")

# Alternative approach - check if AOTriton registers itself with PyTorch
print("\n=== PyTorch Integration Check ===")
try:
    # Check if AOTriton has registered any ops with PyTorch
    if hasattr(torch.ops, 'aotriton'):
        print("torch.ops.aotriton exists")
        all_aotriton_ops = dir(torch.ops.aotriton)
        print(f"Available ops: {all_aotriton_ops}")
        
        # Look for attention-related ops
        attention_ops = [op for op in all_aotriton_ops if 'attn' in op.lower() or 'attention' in op.lower()]
        if attention_ops:
            print(f"Attention ops: {attention_ops}")
            
            # Try to use the first attention op
            if len(attention_ops) > 0:
                op_name = attention_ops[0]
                try:
                    op = getattr(torch.ops.aotriton, op_name)
                    print(f"Found op: {op}")
                    print(f"Op signature: {op}")
                except Exception as e:
                    print(f"Could not access op: {e}")
    else:
        print("torch.ops.aotriton not found")
        
except Exception as e:
    print(f"Error checking PyTorch integration: {e}")

# Check if we need to manually register AOTriton
print("\n=== Manual Registration Check ===")
try:
    # Some libraries require manual registration
    if hasattr(pyaotriton, 'register') or hasattr(pyaotriton, 'init'):
        print("Found registration functions in pyaotriton")
    
    # Check for initialization functions
    if hasattr(pyaotriton.v2, 'init') or hasattr(pyaotriton.v2, 'register'):
        print("Found registration functions in pyaotriton.v2")
        
except Exception as e:
    print(f"Error checking registration: {e}")
