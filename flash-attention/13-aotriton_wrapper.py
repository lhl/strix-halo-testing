# aotriton_wrapper.py
import torch
import pyaotriton
from pyaotriton.v2.flash import attn_fwd, FwdExtraArguments
from triton.testing import do_bench

class AOTritonFlashAttention:
    def __init__(self):
        self.stream = pyaotriton.Stream()
        
    def tensor_to_aotriton(self, tensor):
        """Convert PyTorch tensor to AOTriton tensor."""
        if tensor is None:
            return None
            
        # Create HipMemory from tensor data pointer
        memory = pyaotriton.HipMemory(
            ptr=tensor.data_ptr(),
            size=tensor.numel() * tensor.element_size()
        )
        
        # Determine the correct dtype
        if tensor.dtype == torch.float16:
            dtype = pyaotriton.kFloat16
        elif tensor.dtype == torch.bfloat16:
            dtype = pyaotriton.kBFloat16
        elif tensor.dtype == torch.float32:
            dtype = pyaotriton.kFloat32
        else:
            raise ValueError(f"Unsupported dtype: {tensor.dtype}")
        
        # Create AOTriton tensor with shape info
        shape = list(tensor.shape)
        stride = list(tensor.stride())
        
        # AOTriton uses T4 for 4D tensors
        if len(shape) == 4:
            return pyaotriton.T4(memory, dtype, shape, stride)
        elif len(shape) == 2:
            return pyaotriton.T2(memory, dtype, shape, stride)
        elif len(shape) == 1:
            return pyaotriton.T1(memory, dtype, shape, stride)
        elif len(shape) == 0:
            return pyaotriton.T0(memory, dtype)
        else:
            raise ValueError(f"Unsupported tensor dimension: {len(shape)}")
    
    def forward(self, q, k, v, causal=True, sm_scale=None):
        """Run AOTriton Flash Attention forward pass."""
        
        B, H, S, D = q.shape
        
        # Create output tensor
        out = torch.empty_like(q)
        
        # Create softmax_lse tensor for storing max values
        softmax_lse = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
        
        # Convert tensors to AOTriton format
        q_aot = self.tensor_to_aotriton(q)
        k_aot = self.tensor_to_aotriton(k)
        v_aot = self.tensor_to_aotriton(v)
        out_aot = self.tensor_to_aotriton(out)
        lse_aot = self.tensor_to_aotriton(softmax_lse)
        
        # Calculate scale
        if sm_scale is None:
            sm_scale = 1.0 / (D ** 0.5)
        
        # Create extra arguments
        extargs = FwdExtraArguments()
        
        # Call AOTriton
        try:
            # AOTriton expects specific argument order and types
            result = attn_fwd(
                q=q_aot,
                k=k_aot,
                v=v_aot,
                b=None,  # bias
                sm_scale=sm_scale,
                softmax_lse=lse_aot,
                out=out_aot,
                dropout_p=0.0,
                philox_seed=None,
                philox_offset1=None,
                philox_offset2=0,
                philox_seed_output=None,
                philox_offset_output=None,
                encoded_softmax=None,
                is_causal=causal,
                atomic_for_causal=None,
                stream=self.stream,
                extargs=extargs
            )
            
            # Synchronize
            pyaotriton.hipDeviceSynchronize()
            
            if result != pyaotriton.hipSuccess:
                raise RuntimeError(f"AOTriton returned error code: {result}")
            
            return out
            
        except Exception as e:
            print(f"AOTriton call failed: {e}")
            raise

# Test the wrapper
def test_aotriton_wrapper():
    print("=== Testing AOTriton Wrapper ===")
    
    device = 'cuda'
    dtype = torch.float16
    
    # Test sizes
    B, H, S, D = 2, 8, 1024, 64
    
    # Create tensors
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    # Initialize wrapper
    aot_attn = AOTritonFlashAttention()
    
    print(f"Testing with shape: {q.shape}")
    
    try:
        # Run AOTriton
        output = aot_attn.forward(q, k, v, causal=True)
        print(f"AOTriton output shape: {output.shape}")
        
        # Compare with PyTorch SDPA for correctness
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            ref_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Check if outputs are close
        max_diff = torch.max(torch.abs(output - ref_output))
        print(f"Max difference from reference: {max_diff}")
        
        if max_diff < 0.1:  # Reasonable tolerance for fp16
            print("✓ Correctness check passed!")
        else:
            print("✗ Outputs differ significantly")
        
        # Benchmark
        print("\nBenchmarking...")
        
        # Warmup
        for _ in range(10):
            _ = aot_attn.forward(q, k, v, causal=True)
        
        # Benchmark AOTriton
        aot_time = do_bench(lambda: aot_attn.forward(q, k, v, causal=True))
        
        # Benchmark PyTorch Math backend
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            math_time = do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True))
        
        # Calculate FLOPS
        flops = 4 * B * H * S * S * D
        aot_tflops = flops / (aot_time * 1e9)
        math_tflops = flops / (math_time * 1e9)
        
        print(f"AOTriton: {aot_time:.3f} ms ({aot_tflops:.2f} TFLOPS)")
        print(f"PyTorch Math: {math_time:.3f} ms ({math_tflops:.2f} TFLOPS)")
        print(f"Speedup: {math_time/aot_time:.2f}x")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_aotriton_wrapper()
