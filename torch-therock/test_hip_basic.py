#!/usr/bin/env python3
"""
Basic HIP/ROCm test without PyTorch to verify GPU functionality
"""

import subprocess
import os

def test_rocm_smi():
    """Test ROCm SMI"""
    print("=== ROCm SMI Test ===")
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ROCm SMI working")
            print(result.stdout)
        else:
            print("✗ ROCm SMI failed")
            print(result.stderr)
    except Exception as e:
        print(f"✗ ROCm SMI error: {e}")

def test_hip_info():
    """Test HIP device info"""
    print("\n=== HIP Device Info Test ===")
    hip_code = """
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    
    if (error != hipSuccess) {
        std::cout << "HIP error: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Number of HIP devices: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
    
    return 0;
}
"""
    
    # Write HIP test code
    with open('hip_test.cpp', 'w') as f:
        f.write(hip_code)
    
    # Try to compile and run
    try:
        # Compile using bash to avoid readline conflicts
        compile_cmd = ['bash', '-c', 'hipcc hip_test.cpp -o hip_test']
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("✗ HIP compilation failed:")
            print(result.stderr)
            return
        
        print("✓ HIP code compiled successfully")
        
        # Run
        result = subprocess.run(['bash', '-c', './hip_test'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ HIP device info:")
            print(result.stdout)
        else:
            print("✗ HIP runtime failed:")
            print(result.stderr)
            
    except Exception as e:
        print(f"✗ HIP test error: {e}")
    finally:
        # Cleanup
        for f in ['hip_test.cpp', 'hip_test']:
            if os.path.exists(f):
                os.remove(f)

def test_simple_kernel():
    """Test a simple HIP kernel"""
    print("\n=== Simple HIP Kernel Test ===")
    kernel_code = """
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1024;
    const int size = n * sizeof(float);
    
    // Host vectors
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate device memory
    hipError_t error;
    error = hipMalloc(&d_a, size);
    if (error != hipSuccess) {
        std::cout << "hipMalloc d_a failed: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    error = hipMalloc(&d_b, size);
    if (error != hipSuccess) {
        std::cout << "hipMalloc d_b failed: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    error = hipMalloc(&d_c, size);
    if (error != hipSuccess) {
        std::cout << "hipMalloc d_c failed: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    // Copy data to device
    error = hipMemcpy(d_a, h_a.data(), size, hipMemcpyHostToDevice);
    if (error != hipSuccess) {
        std::cout << "hipMemcpy d_a failed: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    error = hipMemcpy(d_b, h_b.data(), size, hipMemcpyHostToDevice);
    if (error != hipSuccess) {
        std::cout << "hipMemcpy d_b failed: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    hipLaunchKernelGGL(vectorAdd, numBlocks, threadsPerBlock, 0, 0, d_a, d_b, d_c, n);
    error = hipGetLastError();
    if (error != hipSuccess) {
        std::cout << "Kernel launch failed: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    // Wait for kernel to complete
    error = hipDeviceSynchronize();
    if (error != hipSuccess) {
        std::cout << "hipDeviceSynchronize failed: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    // Copy result back to host
    error = hipMemcpy(h_c.data(), d_c, size, hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
        std::cout << "hipMemcpy result failed: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    // Verify result
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) {
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "✓ Vector addition kernel executed successfully!" << std::endl;
        std::cout << "Result: " << h_a[0] << " + " << h_b[0] << " = " << h_c[0] << std::endl;
    } else {
        std::cout << "✗ Vector addition kernel failed - incorrect results" << std::endl;
    }
    
    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    
    return success ? 0 : 1;
}
"""
    
    # Write kernel test code
    with open('kernel_test.cpp', 'w') as f:
        f.write(kernel_code)
    
    try:
        # Compile using bash to avoid readline conflicts
        compile_cmd = ['bash', '-c', 'hipcc kernel_test.cpp -o kernel_test']
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("✗ Kernel compilation failed:")
            print(result.stderr)
            return
        
        print("✓ Kernel code compiled successfully")
        
        # Run
        result = subprocess.run(['bash', '-c', './kernel_test'], capture_output=True, text=True)
        print("Kernel execution result:")
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
            
    except Exception as e:
        print(f"✗ Kernel test error: {e}")
    finally:
        # Cleanup
        for f in ['kernel_test.cpp', 'kernel_test']:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    print("=== Basic ROCm/HIP Functionality Test ===")
    test_rocm_smi()
    test_hip_info()
    test_simple_kernel()