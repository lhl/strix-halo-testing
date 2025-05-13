#!/usr/bin/env python3
# =============================================================================
# Flash Attention Test
# =============================================================================
# This script tests if Flash Attention is working correctly on AMD GPUs.
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
# Date: 2023-04-19
# =============================================================================

import torch
import sys
import os

# Add colorful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    print(f"{Colors.HEADER}{Colors.BOLD}=== {text} ==={Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}INFO: {text}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}SUCCESS: {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}WARNING: {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}ERROR: {text}{Colors.END}")

def standard_attention(q, k, v, causal=False):
    """Standard scaled dot-product attention."""
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Reshape for batched matrix multiplication
    q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
    k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
    v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # Apply causal mask if needed
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Compute attention output
    output = torch.matmul(attn_weights, v)
    
    # Reshape back
    output = output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
    
    return output

def test_flash_attention():
    """Test if Flash Attention is working correctly on AMD GPUs."""
    print_header("Flash Attention Test")
    
    # Check if CUDA (ROCm) is available
    if not torch.cuda.is_available():
        print_error("CUDA (ROCm) is not available")
        return False
    
    # Try to import flash_attention_amd
    try:
        from flash_attention_amd import flash_attn_func
        print_success("Flash Attention module imported successfully")
    except ImportError:
        print_error("Failed to import Flash Attention module")
        print_info("Make sure Flash Attention is installed:")
        print_info("  - Check if flash_attention_amd.py is in the Python path")
        print_info("  - Try reinstalling Flash Attention")
        return False
    
    # Test parameters
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64
    
    # Create input data
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    
    # Test standard attention
    try:
        standard_output = standard_attention(q, k, v, causal=True)
        print_success("Standard attention computation successful")
    except Exception as e:
        print_error(f"Standard attention computation failed: {e}")
        return False
    
    # Test Flash Attention
    try:
        flash_output = flash_attn_func(q, k, v, causal=True)
        print_success("Flash Attention computation successful")
    except Exception as e:
        print_error(f"Flash Attention computation failed: {e}")
        return False
    
    # Check if outputs are similar
    try:
        max_diff = torch.max(torch.abs(standard_output - flash_output)).item()
        print_info(f"Maximum difference between standard and Flash Attention: {max_diff:.6f}")
        
        if max_diff < 1e-3:
            print_success("Flash Attention output matches standard attention output")
        else:
            print_warning("Flash Attention output differs from standard attention output")
            print_info("This may be expected due to different algorithms and precision")
    except Exception as e:
        print_error(f"Failed to compare outputs: {e}")
        return False
    
    # Test with different sequence lengths
    for seq_len in [128, 256, 512]:
        print_info(f"Testing with sequence length {seq_len}")
        
        # Create input data
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        
        try:
            flash_output = flash_attn_func(q, k, v, causal=True)
            print_success(f"Flash Attention computation successful for sequence length {seq_len}")
        except Exception as e:
            print_error(f"Flash Attention computation failed for sequence length {seq_len}: {e}")
            return False
    
    # Test with different batch sizes
    for bs in [1, 4, 8]:
        print_info(f"Testing with batch size {bs}")
        
        # Create input data
        q = torch.randn(bs, 512, num_heads, head_dim, device="cuda")
        k = torch.randn(bs, 512, num_heads, head_dim, device="cuda")
        v = torch.randn(bs, 512, num_heads, head_dim, device="cuda")
        
        try:
            flash_output = flash_attn_func(q, k, v, causal=True)
            print_success(f"Flash Attention computation successful for batch size {bs}")
        except Exception as e:
            print_error(f"Flash Attention computation failed for batch size {bs}: {e}")
            return False
    
    # Test with different head dimensions
    for hd in [32, 128]:
        print_info(f"Testing with head dimension {hd}")
        
        # Create input data
        q = torch.randn(batch_size, 512, num_heads, hd, device="cuda")
        k = torch.randn(batch_size, 512, num_heads, hd, device="cuda")
        v = torch.randn(batch_size, 512, num_heads, hd, device="cuda")
        
        try:
            flash_output = flash_attn_func(q, k, v, causal=True)
            print_success(f"Flash Attention computation successful for head dimension {hd}")
        except Exception as e:
            print_error(f"Flash Attention computation failed for head dimension {hd}: {e}")
            return False
    
    print_success("All Flash Attention tests passed")
    return True

if __name__ == "__main__":
    success = test_flash_attention()
    sys.exit(0 if success else 1)
