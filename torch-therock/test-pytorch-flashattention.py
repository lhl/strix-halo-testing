#!/usr/bin/env python3

import torch
import os

print('=== PyTorch Installation Check ===')
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch ROCm version: {torch.version.hip}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name()}')

print()
print('=== Flash Attention Support Check ===')
try:
    # Check if flash attention is available
    from torch.nn.attention import SDPBackend
    import torch.nn.functional as F
    
    # Check available backends
    backends = torch.backends.cuda.sdp_kernel()
    print(f'Available SDP backends: {backends}')
    
    # Try to use Flash Attention
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        print('Flash Attention backend enabled')
        
        # Create test tensors
        batch_size, seq_len, head_dim = 2, 128, 64
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        q = torch.randn(batch_size, 8, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, 8, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, 8, seq_len, head_dim, device=device, dtype=torch.float16)
        
        print(f'Test tensors created on {device}')
        
        # Run scaled dot product attention
        output = F.scaled_dot_product_attention(q, k, v)
        print(f'Flash Attention test successful! Output shape: {output.shape}')
        
except Exception as e:
    print(f'Flash Attention test failed: {e}')

print()
print('=== AOTriton Check ===')
try:
    import pyaotriton
    print(f'AOTriton version: {pyaotriton.__version__}')
    print('AOTriton imported successfully')
except ImportError as e:
    print(f'AOTriton not available: {e}')
except Exception as e:
    print(f'AOTriton error: {e}')

print()
print('=== Environment Variables ===')
rocm_vars = ['ROCM_PATH', 'HIP_PATH', 'HIP_PLATFORM', 'HIP_ARCH', 'HSA_OVERRIDE_GFX_VERSION']
for var in rocm_vars:
    print(f'{var}: {os.environ.get(var, "Not set")}')

print()
print('=== Testing GFX Version Override ===')
# Try setting HSA_OVERRIDE_GFX_VERSION to map gfx1151 to gfx1100
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
print('Set HSA_OVERRIDE_GFX_VERSION=11.0.0 to test gfx110x mapping')

try:
    # Test with override
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        q = torch.randn(1, 1, 32, 64, device=device, dtype=torch.float16)
        k = torch.randn(1, 1, 32, 64, device=device, dtype=torch.float16)
        v = torch.randn(1, 1, 32, 64, device=device, dtype=torch.float16)
        
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            print('✓ Flash Attention worked with GFX override!')
except Exception as e:
    print(f'✗ Flash Attention still failed with override: {e}')
    
# Reset the environment variable
del os.environ['HSA_OVERRIDE_GFX_VERSION']

print()
print('=== Flash Attention Backend Detection ===')
try:
    # More detailed backend checking
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        # Check what backends are actually available for this device
        q = torch.randn(1, 1, 16, 32, device=device, dtype=torch.float16)
        k = torch.randn(1, 1, 16, 32, device=device, dtype=torch.float16) 
        v = torch.randn(1, 1, 16, 32, device=device, dtype=torch.float16)
        
        # Test each backend individually
        backends_to_test = [
            ('flash', {'enable_flash': True, 'enable_math': False, 'enable_mem_efficient': False}),
            ('mem_efficient', {'enable_flash': False, 'enable_math': False, 'enable_mem_efficient': True}),
            ('math', {'enable_flash': False, 'enable_math': True, 'enable_mem_efficient': False})
        ]
        
        for backend_name, kwargs in backends_to_test:
            try:
                with torch.backends.cuda.sdp_kernel(**kwargs):
                    output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    print(f'✓ {backend_name} backend works')
            except Exception as e:
                print(f'✗ {backend_name} backend failed: {e}')
    else:
        print('CUDA not available, skipping backend tests')
        
except Exception as e:
    print(f'Backend detection failed: {e}')