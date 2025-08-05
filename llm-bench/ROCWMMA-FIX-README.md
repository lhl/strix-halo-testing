# rocWMMA Compatibility Fix for llama.cpp

This directory contains scripts to apply rocWMMA compatibility fixes to llama.cpp checkouts until the fixes can be upstreamed.

## Problem

When building llama.cpp with rocWMMA support (`-DGGML_HIP_ROCWMMA_FATTN=ON`), compilation fails due to warp synchronization mask type conflicts:

- ROCm/rocWMMA headers require 64-bit masks for `__shfl_*_sync` functions
- CUDA-style code in llama.cpp uses 32-bit masks (`0xFFFFFFFF`)
- This causes compilation errors about mask size requirements

## Solution

The fix introduces a conditional macro `GGML_HIP_WARP_MASK` that:
- Expands to `0xFFFFFFFF` for regular HIP builds (no change in behavior)
- Expands to `0xFFFFFFFFFFFFFFFFULL` for rocWMMA builds (64-bit compatibility)

## Scripts

### `apply-rocwmma-fix.sh`

Applies the rocWMMA compatibility fixes to a llama.cpp checkout.

**Usage:**
```bash
./apply-rocwmma-fix.sh /path/to/llama.cpp
```

**What it does:**
1. Modifies `ggml/src/ggml-cuda/vendors/hip.h` to add conditional `GGML_HIP_WARP_MASK` macro
2. Replaces hardcoded `0xFFFFFFFF`/`0xffffffff` with `GGML_HIP_WARP_MASK` in CUDA files
3. Creates backup files for safety
4. Verifies the changes were applied correctly

### `revert-rocwmma-fix.sh`

Reverts the rocWMMA compatibility fixes using backup files.

**Usage:**
```bash
./revert-rocwmma-fix.sh /path/to/llama.cpp
```

**Requirements:**
- Backup files (`.backup`) must exist from the apply script

### `test-rocwmma-fix.sh`

Tests whether the rocWMMA compatibility fixes are properly applied.

**Usage:**
```bash
./test-rocwmma-fix.sh /path/to/llama.cpp
```

**What it checks:**
1. Vendor header contains the conditional macro
2. CUDA files use the macro instead of hardcoded values
3. CMake configuration works with rocWMMA enabled (if cmake available)

## Example Workflow

```bash
# Clone a fresh llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Apply the rocWMMA fixes
../llm-bench/apply-rocwmma-fix.sh .

# Test the fixes
../llm-bench/test-rocwmma-fix.sh .

# Build with rocWMMA support
cmake -B build -S . -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1151" -DGGML_HIP_ROCWMMA_FATTN=ON
cmake --build build --config Release -j$(nproc)

# If you need to revert
../llm-bench/revert-rocwmma-fix.sh .
```

## Technical Details

### Files Modified

**`ggml/src/ggml-cuda/vendors/hip.h`:**
```c
#ifdef GGML_HIP_ROCWMMA_FATTN
// ROCm requires 64-bit masks for __shfl_*_sync functions
#define GGML_HIP_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#else
#define __shfl_sync(mask, var, laneMask, width) __shfl(var, laneMask, width)
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#define GGML_HIP_WARP_MASK 0xFFFFFFFF
#endif
```

**CUDA files (`.cu`, `.cuh`):**
- All instances of `0xFFFFFFFF` and `0xffffffff` replaced with `GGML_HIP_WARP_MASK`

### Compatibility

- ✅ **Regular HIP builds**: No behavior change (`GGML_HIP_WARP_MASK` = `0xFFFFFFFF`)
- ✅ **Regular CUDA builds**: Not affected (uses original macros)
- ✅ **rocWMMA builds**: Uses 64-bit masks (`GGML_HIP_WARP_MASK` = `0xFFFFFFFFFFFFFFFFULL`)

### Tested With

- ROCm 6.3.0+
- AMD Ryzen AI Max+ 395 (gfx1151)
- llama.cpp main branch (January 2025)

## Notes

- This is a temporary fix until the changes can be upstreamed to llama.cpp
- The fix is designed to be minimally invasive and preserve all existing functionality
- Backup files are created automatically for easy reverting
- The scripts include safety checks and verification steps

## Troubleshooting

**"No CUDA files found"**: Some llama.cpp versions may not have CUDA files in the expected location. This is usually fine - the vendor header fix alone may be sufficient.

**"CMake configuration failed"**: This could indicate missing ROCm installation or other system issues. The fix may still be correct.

**"Backup files not found"**: The revert script requires backup files created by the apply script. Use git to restore files manually if needed.

## Contributing

If you find issues with these scripts or have improvements, please:
1. Test thoroughly on different llama.cpp versions
2. Update the compatibility notes above
3. Consider contributing the fixes upstream to llama.cpp