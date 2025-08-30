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
    return 0;
}