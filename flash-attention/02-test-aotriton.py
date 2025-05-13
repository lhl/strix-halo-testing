import torch
import triton
import triton.language as tl

# Test if Triton is using AOTriton backend
print(f"Triton version: {triton.__version__}")

# Try to find backend information differently
if hasattr(triton.runtime, 'driver'):
    print(f"Driver info: {dir(triton.runtime.driver)}")

# Try importing pyaotriton directly
try:
    import pyaotriton
    print("PyAOTriton imported successfully!")
    print(f"PyAOTriton contents: {dir(pyaotriton)}")
except ImportError as e:
    print(f"Cannot import pyaotriton: {e}")

# Check available devices and backends
if hasattr(triton.runtime, 'backend'):
    print(f"Backends: {dir(triton.runtime.backend)}")

# Simple kernel to test
@triton.jit
def vector_add(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# Test on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if device != 'cpu':
    n_elements = 1024
    x = torch.rand(n_elements, device=device)
    y = torch.rand(n_elements, device=device)
    output = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    try:
        vector_add[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        print(f"Kernel executed successfully: {torch.allclose(output, x + y)}")
    except Exception as e:
        print(f"Kernel execution failed: {e}")
else:
    print("No GPU available for testing")

# Check if AOTriton is being used
import os
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
