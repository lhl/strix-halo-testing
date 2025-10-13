# test_attention_benchmark_small.py
import torch
import torch.nn.functional as F
from triton.testing import do_bench
from tabulate import tabulate
import os

# Set environment variables for AOTriton
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'

# Table formatting: override with env var TABLE_FORMAT (e.g., 'simple', 'plain', 'github', 'tsv')
TABLE_FORMAT = os.environ.get('TABLE_FORMAT', 'simple')

torch.set_default_device("cuda")
torch.manual_seed(0)

data_type = torch.float16

def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")

def available_sdpa_backends():
    backends = []
    SDP = torch.nn.attention.SDPBackend
    # Prefer common names, fall back where necessary
    if hasattr(SDP, "FLASH_ATTENTION"):
        backends.append(("FlashAttention (FA2)", SDP.FLASH_ATTENTION))
    if hasattr(SDP, "EFFICIENT_ATTENTION"):
        backends.append(("EfficientAttention", SDP.EFFICIENT_ATTENTION))
    elif hasattr(SDP, "MEM_EFFICIENT_ATTENTION"):
        backends.append(("EfficientAttention", SDP.MEM_EFFICIENT_ATTENTION))
    if hasattr(SDP, "MATH"):
        backends.append(("Math", SDP.MATH))
    return backends

def bench_fwd(impl):
    return do_bench(impl)

def bench_fwd_bwd(impl, qkv, grad_out):
    # Each iteration recreates a fresh graph and avoids grad accumulation
    def _step():
        q, k, v = [t.detach().requires_grad_() for t in qkv]
        out = impl(q, k, v)
        out.backward(grad_out)
    return do_bench(_step)

def sdpa_effective_pairs(S: int, causal: bool) -> int:
    # Number of pairwise score computations per row accounting for causal mask
    if causal:
        return S * (S + 1) // 2
    return S * S

def sdpa_flops(B: int, H: int, S: int, D: int, causal: bool, count_fma_as_2: bool = True):
    # Approximate FLOPs dominated by GEMM-like ops; ignores softmax/exponentials
    # Forward: QK^T and P@V => ~4 * B * H * D * S_eff
    # Backward (matmul terms): dV, dP, dQ, dK => ~8 * B * H * D * S_eff
    s_eff = sdpa_effective_pairs(S, causal)
    unit = 2 if count_fma_as_2 else 1
    fwd = unit * 4 * B * H * D * s_eff
    bwd = unit * 8 * B * H * D * s_eff
    return fwd, bwd, fwd + bwd

def to_tflops(flops: float, time_ms: float) -> float:
    if isinstance(time_ms, str):
        return float('nan')
    secs = time_ms * 1e-3
    if secs <= 0:
        return float('inf')
    return flops / secs / 1e12

def test_attention_sizes(sizes_to_test):
    """Test SDPA backends with causal/non-causal across sizes."""
    results_all = []

    # Determine available backends once
    backends = available_sdpa_backends()
    if not backends:
        print("No SDPA backends detected; exiting.")
        return []

    for name, (B, H, S, D) in sizes_to_test.items():
        print_header(f"Testing {name}: B={B}, H={H}, S={S}, D={D}")

        # Dtype-aware memory requirement
        bytes_per_el = torch.tensor((), dtype=data_type).element_size()
        memory_per_tensor = B * H * S * D * bytes_per_el
        total_memory = memory_per_tensor * 3  # Q, K, V
        print(f"Estimated memory per QKV tensor: {memory_per_tensor / (1024**3):.2f} GB")
        print(f"Total QKV memory: {total_memory / (1024**3):.2f} GB")

        try:
            # Base tensors reused across iterations; we detach inside the timed step
            qkv_base = [
                torch.randn(B, H, S, D, device="cuda", dtype=data_type, requires_grad=True)
                for _ in range(3)
            ]
            grad_out = torch.randn(B, H, S, D, device="cuda", dtype=data_type)

            results = []

            for backend_name, backend in backends:
                for is_causal in (True, False):
                    label = f"{backend_name} ({'causal' if is_causal else 'non-causal'})"
                    try:
                        with torch.nn.attention.sdpa_kernel(backend):
                            # Define impl using positional q,k,v to ease the fwd_bwd helper
                            def impl(q, k, v, _is_causal=is_causal):
                                return F.scaled_dot_product_attention(q, k, v, is_causal=_is_causal)

                            # Forward pass timing (fixed graph OK)
                            fwd_time = bench_fwd(lambda: impl(*qkv_base))
                            # Backward timing with fresh graph per iter
                            step_time = bench_fwd_bwd(impl, qkv_base, grad_out)

                            fwd_flops, bwd_flops, step_flops = sdpa_flops(B, H, S, D, is_causal)
                            fwd_tflops = to_tflops(fwd_flops, fwd_time)
                            step_tflops = to_tflops(step_flops, step_time)

                            results.append([
                                label,
                                fwd_time,
                                fwd_tflops,
                                step_time,
                                step_tflops,
                            ])
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Error with {label}: {e}")
                        results.append([label, "ERROR", "-", "ERROR", "-"])

            floatfmt_spec = (None, ".3f", ".2f", ".3f", ".2f")
            print(tabulate(
                results,
                headers=["Operation", "FW Time (ms)", "FW TFLOPS", "Step Time (ms)", "Step TFLOPS"],
                tablefmt=TABLE_FORMAT,
                floatfmt=floatfmt_spec,
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
        for label, fw_ms, fw_tflops, step_ms, step_tflops in result_list:
            if isinstance(fw_ms, (int, float)):
                print(f"  {label}: FW {fw_ms:.3f} ms ({fw_tflops:.2f} TF/s), Step {step_ms:.3f} ms ({step_tflops:.2f} TF/s)")
            else:
                print(f"  {label}: FW {fw_ms} ms ({fw_tflops} TF/s), Step {step_ms} ms ({step_tflops} TF/s)")
    
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
