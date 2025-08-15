#!/usr/bin/env python3
"""
Test script to verify AOTriton integration with/without PyTorch
This script can test AOTriton standalone or with PyTorch integration
"""

import os
import sys
import argparse

# Set environment for AOTriton
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1151'
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test AOTriton integration')
parser.add_argument('--aotriton-only', action='store_true', 
                   help='Test only AOTriton without PyTorch integration')
parser.add_argument('--torch-only', action='store_true',
                   help='Test only PyTorch integration (requires torch)')
args = parser.parse_args()

# Try to import torch, but don't fail if it's not available
torch_available = False
try:
    import torch
    torch_available = True
except ImportError:
    if not args.aotriton_only:
        print("⚠ PyTorch not available. Use --aotriton-only to test AOTriton standalone.")
        if args.torch_only:
            print("✗ Cannot run torch-only tests without PyTorch installed.")
            sys.exit(1)

print("=== AOTriton Test Suite ===")

# PyTorch information section (only if torch is available and not aotriton-only)
if torch_available and not args.aotriton_only:
    print("=== PyTorch Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"ROCm/HIP version: {torch.version.hip}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device capability: {torch.cuda.get_device_capability(0)}")
elif args.aotriton_only:
    print("=== AOTriton Standalone Mode ===")
    print("Skipping PyTorch information (--aotriton-only specified)")

print("\n=== AOTriton Module Check ===")
try:
    import pyaotriton
    print(f"✓ pyaotriton imported successfully")
    print(f"  Module location: {pyaotriton.__file__}")
    
    if hasattr(pyaotriton, 'v2'):
        print(f"  v2 submodule available: {dir(pyaotriton.v2)}")
        if hasattr(pyaotriton.v2, 'flash'):
            print(f"  ✓ Flash attention module found")
        else:
            print(f"  ✗ Flash attention module NOT found")
    else:
        print(f"  ✗ v2 submodule NOT available")
        
except ImportError as e:
    print(f"✗ pyaotriton import failed: {e}")
    sys.exit(1)

# PyTorch AOTriton ops check (only if torch is available and not aotriton-only)
if torch_available and not args.aotriton_only:
    print("\n=== PyTorch AOTriton Ops Check ===")
    if hasattr(torch.ops, 'aotriton'):
        print(f"✓ torch.ops.aotriton is available")
        aotriton_ops = dir(torch.ops.aotriton)
        print(f"  Available ops: {aotriton_ops}")
        
        # Check for specific attention operations
        attention_ops = [op for op in aotriton_ops if 'attention' in op.lower() or 'flash' in op.lower()]
        if attention_ops:
            print(f"  ✓ Found attention operations: {attention_ops}")
        else:
            print(f"  ⚠ No attention operations found in AOTriton ops")
    else:
        print(f"✗ torch.ops.aotriton is NOT available")

# SDPA and integration tests (only if torch is available and not aotriton-only)
if torch_available and not args.aotriton_only:
    print("\n=== SDPA Backend Check ===")
    print(f"Flash SDPA enabled: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"Memory efficient SDPA enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"Math SDPA enabled: {torch.backends.cuda.math_sdp_enabled()}")

    print("\n=== Test SDPA with Different Backends ===")
    device = 'cuda'
    dtype = torch.float16
    b, h, s, d = 2, 8, 512, 64

    q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    k = torch.randn(b, h, s, d, device=device, dtype=dtype)
    v = torch.randn(b, h, s, d, device=device, dtype=dtype)

    # Test 1: Default SDPA (should work with any backend)
    try:
        print("Testing default SDPA...")
        result1 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        print("  ✓ Default SDPA works")
    except Exception as e:
        print(f"  ✗ Default SDPA failed: {e}")

    # Test 2: Force Flash Attention backend
    try:
        print("Testing Flash Attention backend...")
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            result2 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        print("  ✓ Flash Attention backend works")
    except Exception as e:
        print(f"  ✗ Flash Attention backend failed: {e}")

    # Test 3: Force Math backend (fallback)
    try:
        print("Testing Math backend...")
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            result3 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        print("  ✓ Math backend works")
    except Exception as e:
        print(f"  ✗ Math backend failed: {e}")

print("\n=== Direct AOTriton Call Test ===")
try:
    from pyaotriton.v2.flash import attn_fwd
    print("✓ AOTriton flash attention function imported")
    
    # Note: Direct AOTriton calls require specific tensor types and layouts
    # This is mainly to verify the function is available
    print("  AOTriton function signature available")
    
except ImportError as e:
    print(f"✗ AOTriton flash attention import failed: {e}")
except Exception as e:
    print(f"⚠ AOTriton available but direct call setup failed: {e}")

print("\n=== Environment Variables ===")
relevant_vars = [
    'PYTORCH_ROCM_ARCH',
    'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL',
    'AOTRITON_INSTALLED_PREFIX',
    'LD_LIBRARY_PATH',
    'ROCM_PATH'
]

for var in relevant_vars:
    value = os.getenv(var, 'Not set')
    print(f"{var}: {value}")

print("\n=== Summary ===")
if torch_available and not args.aotriton_only:
    if hasattr(torch.ops, 'aotriton') and len([op for op in dir(torch.ops.aotriton) if not op.startswith('_')]) > 3:
        print("✓ AOTriton integration appears successful!")
        print("  PyTorch should be able to use AOTriton for flash attention.")
    else:
        print("⚠ AOTriton integration incomplete.")
        print("  PyTorch will fall back to math backend for attention operations.")
elif args.aotriton_only or not torch_available:
    try:
        import pyaotriton
        print("✓ AOTriton standalone installation verified!")
        print("  AOTriton can be used independently of PyTorch.")
    except ImportError:
        print("✗ AOTriton not properly installed.")
elif args.torch_only and torch_available:
    if hasattr(torch.ops, 'aotriton'):
        print("✓ PyTorch AOTriton integration verified!")
    else:
        print("⚠ PyTorch available but AOTriton integration missing.")

print("Test completed.")