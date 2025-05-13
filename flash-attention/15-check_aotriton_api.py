# check_aotriton_api.py
import torch
import pyaotriton
import inspect

print("=== AOTriton API Check ===")

# Check HipMemory
print("\nHipMemory methods:")
print(dir(pyaotriton.HipMemory))
if hasattr(pyaotriton.HipMemory, '__init__'):
    print(f"HipMemory.__init__ signature: {inspect.signature(pyaotriton.HipMemory.__init__)}")

# Check available tensor types
print("\nTensor types:")
for t in ['T0', 'T1', 'T2', 'T4']:
    if hasattr(pyaotriton, t):
        tensor_type = getattr(pyaotriton, t)
        print(f"{t} methods: {dir(tensor_type)}")
        if hasattr(tensor_type, '__init__'):
            try:
                sig = inspect.signature(tensor_type.__init__)
                print(f"{t}.__init__ signature: {sig}")
            except:
                print(f"Cannot get signature for {t}.__init__")

# Check Stream
print("\nStream methods:")
print(dir(pyaotriton.Stream))
if hasattr(pyaotriton.Stream, '__init__'):
    try:
        print(f"Stream.__init__ signature: {inspect.signature(pyaotriton.Stream.__init__)}")
    except:
        pass

# Check FwdExtraArguments
print("\nFwdExtraArguments methods:")
fwd_args = pyaotriton.v2.flash.FwdExtraArguments
print(dir(fwd_args))

# Let's see if there's a different way to create tensors
print("\n=== Practical Test ===")
try:
    # Create a PyTorch tensor
    torch_tensor = torch.randn(1024, device='cuda', dtype=torch.float16)
    print(f"PyTorch tensor created: shape={torch_tensor.shape}, ptr={torch_tensor.data_ptr()}")
    
    # Try creating HipMemory without parameters
    hip_mem = pyaotriton.HipMemory()
    print(f"Empty HipMemory created: {hip_mem}")
    print(f"HipMemory attributes: {[attr for attr in dir(hip_mem) if not attr.startswith('_')]}")
    
    # Check if we can set properties
    if hasattr(hip_mem, 'ptr'):
        print(f"HipMemory has ptr attribute")
    if hasattr(hip_mem, 'size'):
        print(f"HipMemory has size attribute")
        
    # Try creating a Stream
    stream = pyaotriton.Stream()
    print(f"Stream created: {stream}")
    
    # Check if there are factory functions
    print("\nLooking for factory functions...")
    for name in dir(pyaotriton):
        if 'create' in name.lower() or 'make' in name.lower() or 'from' in name.lower():
            print(f"Found: {name}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Check the documentation
print("\n=== Documentation ===")
print(f"pyaotriton.__doc__: {pyaotriton.__doc__}")
if hasattr(pyaotriton, 'HipMemory'):
    print(f"HipMemory.__doc__: {pyaotriton.HipMemory.__doc__}")
if hasattr(pyaotriton, 'T4'):
    print(f"T4.__doc__: {pyaotriton.T4.__doc__}")
