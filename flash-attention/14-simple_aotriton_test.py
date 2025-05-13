# simple_aotriton_test.py
import torch
import pyaotriton
import ctypes

print("=== Simple AOTriton Test ===")

# Check if we can create AOTriton types
try:
    # Create a simple memory block
    size = 1024
    data = torch.randn(size, device='cuda', dtype=torch.float16)
    
    # Try to create HipMemory directly
    mem = pyaotriton.HipMemory(
        ptr=data.data_ptr(),
        size=data.numel() * data.element_size()
    )
    print(f"Created HipMemory: {mem}")
    
    # Try to create a tensor
    t1 = pyaotriton.T1(mem, pyaotriton.kFloat16, [size], [1])
    print(f"Created T1 tensor: {t1}")
    
except Exception as e:
    print(f"Error creating AOTriton types: {e}")

# Let's check if we can find the actual compiled kernels
print("\n=== Checking for compiled kernels ===")
try:
    # AOTriton might have debug functions we can use
    if hasattr(pyaotriton.v2.flash, 'check_gpu'):
        result = pyaotriton.v2.flash.check_gpu()
        print(f"GPU check result: {result}")
except Exception as e:
    print(f"Error checking GPU: {e}")

# Check if AOTriton has any environment variable requirements
print("\n=== Environment Check ===")
import os
env_vars = ['AOTRITON_RUNTIME_LOG', 'AOTRITON_ENABLE_TUNING', 'AOTRITON_GPU_ARCH']
for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"{var}: {value}")
