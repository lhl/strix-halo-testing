# test_aotriton_direct.py
import torch
import os
import pyaotriton

# Set environment
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1151'
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'

print("=== AOTriton Direct Test ===")
print(f"AOTriton module: {pyaotriton}")
print(f"AOTriton directory: {dir(pyaotriton)}")

# Check for Flash Attention in AOTriton
if hasattr(pyaotriton, 'v2'):
    print(f"\nAOTriton v2 available: {dir(pyaotriton.v2)}")

# Check torch ops
print(f"\n=== Torch Ops ===")
if hasattr(torch.ops, 'aotriton'):
    print(f"torch.ops.aotriton: {dir(torch.ops.aotriton)}")
    # Try to find specific attention ops
    if hasattr(torch.ops.aotriton, 'flash_attention'):
        print("Found flash_attention in aotriton ops!")
    else:
        print("No flash_attention found in aotriton ops")

# Let's try to manually use AOTriton attention if available
print("\n=== Testing AOTriton Functions ===")
device = 'cuda'
dtype = torch.float16
b, h, s, d = 2, 4, 1024, 64

q = torch.randn(b, h, s, d, device=device, dtype=dtype)
k = torch.randn(b, h, s, d, device=device, dtype=dtype)
v = torch.randn(b, h, s, d, device=device, dtype=dtype)

# Try to find and use AOTriton's attention implementation
if hasattr(pyaotriton.v2, 'flash'):
    try:
        from pyaotriton.v2.flash import attn_fwd
        print("Found AOTriton flash attention functions!")
        
        # AOTriton typically needs specific tensor layouts
        # Let's try to call it directly
        try:
            # AOTriton often expects BHSD layout
            output = attn_fwd(q, k, v, None, 0.0, True, None)
            print("Direct AOTriton call successful!")
        except Exception as e:
            print(f"Direct AOTriton call failed: {e}")
    except ImportError:
        print("Could not import AOTriton flash functions")

# Check if AOTriton is registered as a PyTorch backend
print("\n=== Checking PyTorch Backend Registration ===")
try:
    # List all registered ops
    all_ops = torch._C._jit_get_all_schemas()
    attention_ops = [op for op in all_ops if 'attention' in str(op).lower() and 'aotriton' in str(op).lower()]
    
    if attention_ops:
        print(f"Found {len(attention_ops)} AOTriton attention ops:")
        for op in attention_ops[:5]:  # Show first 5
            print(f"  {op}")
    else:
        print("No AOTriton attention ops found in PyTorch registry")
except:
    pass

# Check environment variables that might affect backend selection
print("\n=== Relevant Environment Variables ===")
important_vars = [
    'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL',
    'PYTORCH_ROCM_ARCH',
    'HSA_OVERRIDE_GFX_VERSION',
    'ROCM_ARCH',
    'HIP_VISIBLE_DEVICES',
    'PYTORCH_ROCM_AOTRITON_PREFER_DEFAULT'
]

for var in important_vars:
    print(f"{var}: {os.getenv(var, 'Not set')}")

# Try to understand why PyTorch isn't using AOTriton
print("\n=== Debugging PyTorch SDPA Selection ===")
try:
    # This might help us understand the backend selection logic
    import torch._dynamo.config
    import logging
    
    # Enable more verbose logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Try to get more information about why Flash Attention isn't available
    q_test = torch.randn(1, 1, 64, 64, device='cuda', dtype=torch.float16)
    k_test = torch.randn(1, 1, 64, 64, device='cuda', dtype=torch.float16)
    v_test = torch.randn(1, 1, 64, 64, device='cuda', dtype=torch.float16)
    
    with torch.autograd.profiler.record_function("SDPA_selection"):
        result = torch.nn.functional.scaled_dot_product_attention(q_test, k_test, v_test, is_causal=True)
        
    print("SDPA execution completed")
except Exception as e:
    print(f"Error during SDPA debugging: {e}")

# Try to manually check if the correct architecture is being detected
print("\n=== Architecture Detection ===")
try:
    # Check what PyTorch thinks the architecture is
    if torch.cuda.is_available():
        device_prop = torch.cuda.get_device_properties(0)
        print(f"Device properties: {device_prop}")
        
        # Try to get the GFX version directly
        import subprocess
        try:
            result = subprocess.run(['rocminfo'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'gfx' in line.lower() and 'name' in line.lower():
                    print(f"rocminfo: {line.strip()}")
        except:
            pass
except Exception as e:
    print(f"Error checking architecture: {e}")
