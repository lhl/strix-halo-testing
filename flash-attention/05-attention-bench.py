# test_attention_benchmark_small.py
from functools import lru_cache
import torch
import torch.nn.functional as F
from triton.testing import do_bench
from tabulate import tabulate
import os

# Set environment variables for AOTriton
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'

torch.set_default_device("cuda")
torch.manual_seed(0)

data_type = torch.float16

def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12

def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")

def test_attention_sizes(sizes_to_test):
    """Test different attention implementations with various sizes."""
    
    results_all = []
    
    for name, (B, H, S, D) in sizes_to_test.items():
        print_header(f"Testing {name}: B={B}, H={H}, S={S}, D={D}")
        
        # Check memory requirement
        memory_per_tensor = B * H * S * D * 2  # float16 = 2 bytes
        total_memory = memory_per_tensor * 3  # Q, K, V
        print(f"Estimated memory per QKV tensor: {memory_per_tensor / (1024**3):.2f} GB")
        print(f"Total QKV memory: {total_memory / (1024**3):.2f} GB")
        
        try:
            qkv = [
                torch.randn(B, H, S, D, device="cuda", dtype=data_type, requires_grad=True)
                for _ in range(3)
            ]
            gradOut = torch.randn(B, H, S, D, device="cuda", dtype=data_type)
            
            # Different attention implementations
            causal_fa2 = lambda: F.scaled_dot_product_attention(*qkv, is_causal=True)
            regular_sdpa = lambda: F.scaled_dot_product_attention(*qkv)
            
            # Benchmark
            results = []
            flops = 0.5 * B * H * D * S * S  # Causal attention FLOPS
            
            implementations = [
                ("Causal FA2", causal_fa2),
                ("Regular SDPA", regular_sdpa),
            ]
            
            for impl_name, impl_func in implementations:
                try:
                    # Forward pass
                    fwd_time = do_bench(impl_func)
                    fwd_out = impl_func()
                    
                    # Backward pass
                    bwd_time = do_bench(lambda: fwd_out.backward(gradOut, retain_graph=True))
                    
                    results.append([
                        impl_name,
                        f"{fwd_time:.4f}",
                        f"{calculate_tflops(flops, fwd_time, 4):.2f}",
                        f"{bwd_time:.4f}",
                        f"{calculate_tflops(flops, bwd_time, 10):.2f}",
                    ])
                    
                    del fwd_out
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error with {impl_name}: {e}")
                    results.append([impl_name, "ERROR", "-", "ERROR", "-"])
            
            print(tabulate(
                results,
                headers=["Operation", "FW Time (ms)", "FW FLOPS (TF/s)", "BW Time (ms)", "BW FLOPS (TF/s)"],
                tablefmt="grid",
            ))
            
            results_all.append((name, results))
            
        except Exception as e:
            print(f"Failed to test {name}: {e}")
        
        finally:
            torch.cuda.empty_cache()
        
        print()
    
    return results_all

def check_aotriton_status():
    """Check if AOTriton is properly loaded."""
    print_header("AOTriton Status Check")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.version, 'hip'):
        print(f"ROCm version: {torch.version.hip}")
    
    print(f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: {os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', 'Not set')}")
    
    try:
        import pyaotriton
        print("pyaotriton imported successfully")
        
        if hasattr(torch.ops, 'aotriton'):
            print("torch.ops.aotriton is available")
        else:
            print("torch.ops.aotriton is NOT available")
            
    except ImportError:
        print("Could not import pyaotriton")
    
    print()

if __name__ == "__main__":
    check_aotriton_status()
    
    # Define test sizes - start small and increase
    sizes_to_test = {
        "Tiny": (1, 1, 128, 64),
        "Small": (2, 4, 512, 64),
        "Medium": (4, 8, 1024, 64),
        "Large": (8, 16, 2048, 64),
        # Only test larger sizes if you have enough memory
        "XLarge": (16, 16, 4096, 64),
    }
    
    results = test_attention_sizes(sizes_to_test)
    
    # Summary
    print_header("Summary")
    for name, result_list in results:
        print(f"{name}:")
        for result in result_list:
            print(f"  {result[0]}: {result[1]} ms")
    
    # Test with different dtypes
    print_header("Testing with different dtypes")
    
    for dtype in [torch.float16, torch.bfloat16]:
        print(f"\nTesting with {dtype}")
        try:
            q = torch.randn(1, 1, 128, 64, device="cuda", dtype=dtype)
            k = torch.randn(1, 1, 128, 64, device="cuda", dtype=dtype)
            v = torch.randn(1, 1, 128, 64, device="cuda", dtype=dtype)
            
            result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            print(f"Success with {dtype}")
        except Exception as e:
            print(f"Error with {dtype}: {e}")
