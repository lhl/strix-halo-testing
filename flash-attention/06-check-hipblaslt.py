# check_hipblaslt.py
import torch
import os
import subprocess

print("=== Environment ===")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"PYTORCH_ROCM_ARCH: {os.environ.get('PYTORCH_ROCM_ARCH', 'Not set')}")
print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'Not set')}")

print("\n=== PyTorch Info ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.version, 'hip'):
    print(f"ROCm version: {torch.version.hip}")

print("\n=== GPU Info ===")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Architecture: {torch.cuda.get_arch_list()}")
    
    # Get actual GPU architecture
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'gfx' in line.lower():
                print(f"rocminfo: {line.strip()}")
    except:
        pass

print("\n=== hipBLASLt Check ===")
# Check if hipBLASLt is loaded
try:
    result = subprocess.run(['ldd', '/usr/local/lib64/python3.13/site-packages/torch/lib/libtorch_hip.so'], 
                          capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'hipblaslt' in line.lower():
            print(f"hipBLASLt linking: {line.strip()}")
except:
    pass

# Check hipBLASLt directory
import glob
hipblaslt_path = "/opt/rocm/lib/hipblaslt"
if os.path.exists(hipblaslt_path):
    print(f"\nhipBLASLt directory contents:")
    subdirs = glob.glob(f"{hipblaslt_path}/*")
    for subdir in subdirs:
        if os.path.isdir(subdir):
            print(f"  {os.path.basename(subdir)}/")
            files = glob.glob(f"{subdir}/*.dat")[:5]  # Show first 5 files
            for f in files:
                print(f"    {os.path.basename(f)}")
            if len(glob.glob(f"{subdir}/*.dat")) > 5:
                print(f"    ... and {len(glob.glob(f'{subdir}/*.dat')) - 5} more files")
