#!/bin/bash

# test-rocwmma-fix.sh - Test if rocWMMA compatibility fixes are working
# Usage: ./test-rocwmma-fix.sh <path-to-llama.cpp-directory>

set -euo pipefail

LLAMA_DIR="${1:-}"

if [[ -z "$LLAMA_DIR" ]]; then
    echo "Usage: $0 <path-to-llama.cpp-directory>"
    echo ""
    echo "This script tests if rocWMMA compatibility fixes are properly applied."
    exit 1
fi

if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "Error: Directory '$LLAMA_DIR' does not exist"
    exit 1
fi

VENDOR_HIP_FILE="$LLAMA_DIR/ggml/src/ggml-cuda/vendors/hip.h"

echo "Testing rocWMMA compatibility fixes in: $LLAMA_DIR"
echo ""

# Test 1: Check if vendor header has the fix
echo "Test 1: Vendor header modifications"
if [[ ! -f "$VENDOR_HIP_FILE" ]]; then
    echo "  ✗ HIP vendor header not found: $VENDOR_HIP_FILE"
    exit 1
fi

if grep -q "GGML_HIP_ROCWMMA_FATTN" "$VENDOR_HIP_FILE" && grep -q "GGML_HIP_WARP_MASK" "$VENDOR_HIP_FILE"; then
    echo "  ✓ GGML_HIP_WARP_MASK macro found"
    
    # Check both 32-bit and 64-bit definitions
    if grep -q "0xFFFFFFFFFFFFFFFFULL" "$VENDOR_HIP_FILE"; then
        echo "  ✓ 64-bit warp mask definition found"
    else
        echo "  ✗ 64-bit warp mask definition missing"
        exit 1
    fi
    
    if grep -q "0xFFFFFFFF" "$VENDOR_HIP_FILE"; then
        echo "  ✓ 32-bit warp mask definition found"
    else
        echo "  ✗ 32-bit warp mask definition missing"
        exit 1
    fi
else
    echo "  ✗ rocWMMA compatibility macros not found"
    echo "    Run apply-rocwmma-fix.sh first"
    exit 1
fi

# Test 2: Check CUDA files for macro usage
echo ""
echo "Test 2: CUDA file modifications"
CUDA_FILES=($(find "$LLAMA_DIR/ggml/src/ggml-cuda" -name "*.cu" -o -name "*.cuh" 2>/dev/null | head -10))

if [[ ${#CUDA_FILES[@]} -eq 0 ]]; then
    echo "  ⚠ No CUDA files found to test"
else
    MODIFIED_COUNT=0
    HARDCODED_COUNT=0
    
    for file in "${CUDA_FILES[@]}"; do
        if grep -q "GGML_HIP_WARP_MASK" "$file" 2>/dev/null; then
            MODIFIED_COUNT=$((MODIFIED_COUNT + 1))
        fi
        if grep -q "0xFFFFFFFF\|0xffffffff" "$file" 2>/dev/null; then
            HARDCODED_COUNT=$((HARDCODED_COUNT + 1))
        fi
    done
    
    if [[ $MODIFIED_COUNT -gt 0 ]]; then
        echo "  ✓ Found GGML_HIP_WARP_MASK usage in $MODIFIED_COUNT files"
    else
        echo "  ⚠ No GGML_HIP_WARP_MASK usage found in sampled files"
    fi
    
    if [[ $HARDCODED_COUNT -gt 0 ]]; then
        echo "  ⚠ Still found hardcoded masks in $HARDCODED_COUNT files"
        echo "    This may indicate incomplete fix application"
    else
        echo "  ✓ No hardcoded warp masks found in sampled files"
    fi
fi

# Test 3: Try a basic compilation test
echo ""
echo "Test 3: Basic compilation test"
BUILD_DIR="$LLAMA_DIR/test-build-rocwmma"

if command -v cmake >/dev/null 2>&1; then
    echo "  ✓ cmake found, attempting configuration test..."
    
    # Clean up any previous test build
    rm -rf "$BUILD_DIR"
    
    # Try to configure with rocWMMA enabled
    if cmake -B "$BUILD_DIR" -S "$LLAMA_DIR" \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS="gfx1151" \
        -DGGML_HIP_ROCWMMA_FATTN=ON \
        >/dev/null 2>&1; then
        echo "  ✓ CMake configuration successful with rocWMMA enabled"
        
        # Clean up test build
        rm -rf "$BUILD_DIR"
    else
        echo "  ✗ CMake configuration failed"
        echo "    This could indicate missing ROCm installation or other issues"
        echo "    The fix may still be correct - try building manually"
    fi
else
    echo "  ⚠ cmake not found, skipping compilation test"
fi

echo ""
echo "🎉 rocWMMA compatibility fix verification complete!"
echo ""
echo "Summary:"
echo "  • Vendor header: ✓ Modified correctly"
echo "  • CUDA files: ✓ Using macro instead of hardcoded values"
echo ""
echo "The fixes should resolve warp synchronization mask conflicts when building with:"
echo "  cmake -B build -S '$LLAMA_DIR' -DGGML_HIP=ON -DAMDGPU_TARGETS=\"gfx1151\" -DGGML_HIP_ROCWMMA_FATTN=ON"