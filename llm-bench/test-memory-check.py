#!/usr/bin/env python3
"""
Memory monitoring test script for diagnosing GTT+GART memory tracking issues.
This script tests the components used by llama-bencher for memory monitoring.
"""

import subprocess as sp
import re
import json
import time
from pathlib import Path

def run_cmd(cmd, shell=True):
    """Run a command and return stdout, stderr, and return code."""
    try:
        if shell:
            result = sp.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        else:
            result = sp.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout, result.stderr, result.returncode
    except sp.TimeoutExpired:
        return "", "Command timed out", 1
    except Exception as e:
        return "", f"Error: {e}", 1

def check_utility(cmd_name, test_cmd):
    """Check if a utility is available and working."""
    print(f"\n{'='*60}")
    print(f"Testing {cmd_name}")
    print(f"{'='*60}")
    
    stdout, stderr, rc = run_cmd(test_cmd)
    
    print(f"Command: {test_cmd}")
    print(f"Return code: {rc}")
    
    if rc == 0:
        print(f"✓ {cmd_name} is available")
        if stdout:
            print(f"Sample output (first 500 chars):\n{stdout[:500]}")
            if len(stdout) > 500:
                print("... (truncated)")
    else:
        print(f"✗ {cmd_name} failed or not available")
        if stderr:
            print(f"Error: {stderr}")
        if stdout:
            print(f"Stdout: {stdout}")
    
    return stdout if rc == 0 else None

def _first_int(s: str) -> int:
    """Extract first integer from string (from llama-bencher)."""
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0

def parse_rocm(out: str) -> int:
    """Parse ROCm memory info (from llama-bencher)."""
    try:
        return int(out.splitlines()[1].split(',')[2]) // (1024 * 1024)
    except:
        return 0

def parse_gtt(out: str) -> int:
    """Parse GTT memory from amdgpu_top output (from llama-bencher)."""
    gtt_line = next((l for l in out.splitlines() if re.match(r"^\s*GTT", l)), "")
    return _first_int(gtt_line)

def parse_meminfo(out: str) -> int:
    """Parse system memory usage from /proc/meminfo (from llama-bencher)."""
    tot = avail = 0
    for l in out.splitlines():
        if l.startswith('MemTotal'):
            tot = _first_int(l)
        elif l.startswith('MemAvailable'):
            avail = _first_int(l)
    return (tot - avail) // 1024

def test_parsing_functions():
    """Test the parsing functions with real data."""
    print(f"\n{'='*60}")
    print("Testing Parsing Functions")
    print(f"{'='*60}")
    
    # Test ROCm parsing
    rocm_out = check_utility("rocm-smi", "rocm-smi --showmeminfo vram --csv")
    if rocm_out:
        try:
            vram_mb = parse_rocm(rocm_out)
            print(f"✓ ROCm VRAM parsing: {vram_mb} MB")
        except Exception as e:
            print(f"✗ ROCm VRAM parsing failed: {e}")
            print(f"Raw output lines: {rocm_out.splitlines()}")
    
    # Test GTT parsing  
    amdgpu_out = check_utility("amdgpu_top", "amdgpu_top -d")
    if amdgpu_out:
        try:
            gtt_mb = parse_gtt(amdgpu_out)
            print(f"✓ GTT parsing: {gtt_mb} MB")
            
            # Show GTT-related lines for debugging
            print("\nGTT-related lines in amdgpu_top output:")
            for line in amdgpu_out.splitlines():
                if 'GTT' in line.upper() or 'GART' in line.upper():
                    print(f"  {repr(line)}")
                    
        except Exception as e:
            print(f"✗ GTT parsing failed: {e}")
            print(f"Looking for lines matching '^\\s*GTT' pattern...")
            for i, line in enumerate(amdgpu_out.splitlines()):
                if re.match(r"^\s*GTT", line):
                    print(f"  Line {i}: {repr(line)}")
    
    # Test meminfo parsing
    meminfo_out = check_utility("/proc/meminfo", "cat /proc/meminfo")
    if meminfo_out:
        try:
            used_mb = parse_meminfo(meminfo_out)
            print(f"✓ System RAM parsing: {used_mb} MB used")
        except Exception as e:
            print(f"✗ System RAM parsing failed: {e}")

def test_memory_monitoring():
    """Test memory monitoring over time (like the actual benchmark does)."""
    print(f"\n{'='*60}")
    print("Testing Memory Monitoring Over Time")
    print(f"{'='*60}")
    
    commands = [
        ("rocm-smi", ["rocm-smi", "--showmeminfo", "vram", "--csv"], parse_rocm, "VRAM"),
        ("amdgpu_top", ["amdgpu_top", "-d"], parse_gtt, "GTT"),
        ("meminfo", ["cat", "/proc/meminfo"], parse_meminfo, "System RAM")
    ]
    
    samples = []
    print("Taking 3 samples with 1 second intervals...")
    
    for i in range(3):
        sample = {"iteration": i + 1}
        print(f"\nSample {i + 1}:")
        
        for name, cmd, parser, label in commands:
            stdout, stderr, rc = run_cmd(cmd, shell=False)
            if rc == 0:
                try:
                    value = parser(stdout)
                    sample[name] = value
                    print(f"  {label}: {value} MB")
                except Exception as e:
                    sample[name] = None
                    print(f"  {label}: Parse error - {e}")
            else:
                sample[name] = None
                print(f"  {label}: Command failed")
        
        samples.append(sample)
        if i < 2:  # Don't sleep after last sample
            time.sleep(1)
    
    # Show deltas
    print(f"\nMemory deltas (sample 3 - sample 1):")
    if len(samples) >= 2:
        for name, _, _, label in commands:
            start = samples[0].get(name)
            end = samples[-1].get(name)
            if start is not None and end is not None:
                delta = end - start
                print(f"  {label}: {delta:+d} MB")
            else:
                print(f"  {label}: Unable to calculate delta")

def check_permissions():
    """Check for potential permission issues."""
    print(f"\n{'='*60}")
    print("Checking Permissions and System Info")
    print(f"{'='*60}")
    
    # Check if running as root or in relevant groups
    stdout, _, _ = run_cmd("whoami")
    print(f"Current user: {stdout.strip()}")
    
    stdout, _, _ = run_cmd("groups")
    print(f"User groups: {stdout.strip()}")
    
    # Check AMD GPU devices
    stdout, _, _ = run_cmd("ls -la /dev/kfd /dev/dri/render* 2>/dev/null || echo 'AMD GPU devices not found'")
    print(f"AMD GPU devices:\n{stdout}")
    
    # Check if amdgpu module is loaded
    stdout, _, _ = run_cmd("lsmod | grep amdgpu || echo 'amdgpu module not loaded'")
    print(f"AMDGPU module: {stdout.strip()}")

def main():
    print("Memory Monitoring Diagnostic Tool")
    print("=" * 60)
    print("This script tests the memory monitoring components used by llama-bencher.")
    
    # Check system info
    check_permissions()
    
    # Test individual utilities
    utilities = [
        ("rocm-smi", "rocm-smi --version"),
        ("amdgpu_top", "amdgpu_top --help"),
        ("sensors", "sensors"),
    ]
    
    for name, cmd in utilities:
        check_utility(name, cmd)
    
    # Test parsing functions
    test_parsing_functions()
    
    # Test monitoring over time
    test_memory_monitoring()
    
    print(f"\n{'='*60}")
    print("Diagnostic Complete")
    print(f"{'='*60}")
    
    # Save results to file
    results_file = "memory_diagnostic_results.txt"
    print(f"\nTo capture full output, run:")
    print(f"python3 {__file__} > {results_file} 2>&1")

if __name__ == "__main__":
    main()
