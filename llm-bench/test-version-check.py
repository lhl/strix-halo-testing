#!/usr/bin/env python3
"""
Compare system versions between cluster1 and cluster2 to find GTT reporting differences.
"""

import subprocess as sp
import re

def run_cmd(cmd):
    try:
        return sp.check_output(cmd, shell=True, text=True, stderr=sp.STDOUT).strip()
    except Exception as e:
        return f"Error: {e}"

def check_versions():
    commands = {
        "amdgpu_top_version": "amdgpu_top --version 2>&1 | head -1",
        "rocm_version": "rocminfo | grep -m1 'Runtime Version' | awk '{print $3}' 2>/dev/null || echo 'Not found'",
        "rocm_smi_version": "rocm-smi --version | head -1",
        "kernel": "uname -r",
        "amdgpu_driver": "modinfo amdgpu | grep '^version:' | awk '{print $2}' || echo 'Not found'",
        "gpu_info": "lspci | grep -i 'VGA\\|Display' | head -1",
        "rocm_path": "which rocm-smi",
        "amdgpu_top_path": "which amdgpu_top",
        "libdrm": "pkg-config --modversion libdrm 2>/dev/null || echo 'Not found'",
    }
    
    print("System Version Information:")
    print("=" * 50)
    
    for name, cmd in commands.items():
        result = run_cmd(cmd)
        print(f"{name:20}: {result}")

def test_gtt_alternatives():
    """Test alternative ways to get GTT memory info."""
    print("\n" + "=" * 50)
    print("Alternative GTT Memory Sources:")
    print("=" * 50)
    
    alternatives = {
        "rocm_smi_memory": "rocm-smi --showmeminfo gtt --csv 2>/dev/null || echo 'Not supported'",
        "sysfs_gtt": "find /sys -name '*gtt*' -type f 2>/dev/null | head -5",
        "debugfs_gtt": "ls /sys/kernel/debug/dri/*/amdgpu_gtt* 2>/dev/null || echo 'Not accessible'", 
        "proc_meminfo_huge": "grep -i huge /proc/meminfo",
        "system_mem_vs_available": "free -m | awk 'NR==2{print \"Used:\",$3,\"MB\"; print \"Available:\",$7,\"MB\"}'",
    }
    
    for name, cmd in alternatives.items():
        result = run_cmd(cmd)
        print(f"\n{name}:")
        print(f"  {result}")

def suggest_workarounds():
    """Suggest workarounds for GTT tracking."""
    print("\n" + "=" * 50)
    print("Potential Workarounds:")
    print("=" * 50)
    
    print("""
1. Use system memory delta as proxy:
   - Monitor /proc/meminfo MemAvailable changes
   - Should correlate with GTT usage for large models

2. Try different amdgpu_top options:
   - amdgpu_top -J (JSON output) might be more reliable
   - Check if different flags work better

3. Update amdgpu_top:
   - Build latest version from: https://github.com/Umio-Yasuno/amdgpu_top
   
4. Alternative memory tracking:
   - Use /proc/pid/status for process-specific memory
   - Monitor /sys/kernel/debug/dri/*/amdgpu_gem_info (if accessible)
   
5. Modify llama-bencher to fallback to system memory:
   - When GTT delta is suspiciously low, use system memory delta instead
""")

if __name__ == "__main__":
    check_versions()
    test_gtt_alternatives() 
    suggest_workarounds()
    
    print(f"\n{'='*50}")
    print("Run this on both cluster1 and cluster2, then compare outputs!")
