#!/bin/bash

# revert-rocwmma-fix.sh - Revert rocWMMA compatibility fixes from llama.cpp
# Usage: ./revert-rocwmma-fix.sh <path-to-llama.cpp-directory>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${1:-}"

if [[ -z "$LLAMA_DIR" ]]; then
    echo "Usage: $0 <path-to-llama.cpp-directory>"
    echo ""
    echo "This script reverts the rocWMMA compatibility fixes applied by apply-rocwmma-fix.sh"
    echo ""
    echo "Example:"
    echo "  $0 ./llama.cpp"
    echo "  $0 /path/to/your/llama.cpp"
    exit 1
fi

if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "Error: Directory '$LLAMA_DIR' does not exist"
    exit 1
fi

VENDOR_HIP_FILE="$LLAMA_DIR/ggml/src/ggml-cuda/vendors/hip.h"

if [[ ! -f "$VENDOR_HIP_FILE" ]]; then
    echo "Error: HIP vendor header not found at: $VENDOR_HIP_FILE"
    exit 1
fi

echo "Reverting rocWMMA compatibility fixes from: $LLAMA_DIR"
echo ""

# Check if fixes are applied
if ! grep -q "GGML_HIP_WARP_MASK" "$VENDOR_HIP_FILE" 2>/dev/null; then
    echo "rocWMMA fixes do not appear to be applied (GGML_HIP_WARP_MASK not found)"
    echo "Nothing to revert."
    exit 0
fi

echo "Step 1: Checking for backup files..."

BACKUP_COUNT=0

# Check for vendor header backup
if [[ -f "$VENDOR_HIP_FILE.backup" ]]; then
    echo "  ✓ Found vendor header backup"
    BACKUP_COUNT=$((BACKUP_COUNT + 1))
fi

# Check for CUDA file backups
CUDA_BACKUPS=($(find "$LLAMA_DIR/ggml/src/ggml-cuda" -name "*.backup" 2>/dev/null | sort))
if [[ ${#CUDA_BACKUPS[@]} -gt 0 ]]; then
    echo "  ✓ Found ${#CUDA_BACKUPS[@]} CUDA file backups"
    BACKUP_COUNT=$((BACKUP_COUNT + ${#CUDA_BACKUPS[@]}))
fi

if [[ $BACKUP_COUNT -eq 0 ]]; then
    echo "  ⚠ No backup files found"
    echo ""
    echo "This script can only revert changes if backup files (.backup) exist."
    echo "If you want to revert manually:"
    echo "  1. Use git to restore original files: git checkout -- ggml/src/ggml-cuda/"
    echo "  2. Or re-clone the repository"
    exit 1
fi

echo ""
echo "Step 2: Restoring files from backups..."

# Restore vendor header
if [[ -f "$VENDOR_HIP_FILE.backup" ]]; then
    mv "$VENDOR_HIP_FILE.backup" "$VENDOR_HIP_FILE"
    echo "  ✓ Restored: $(basename "$VENDOR_HIP_FILE")"
fi

# Restore CUDA files
RESTORED_COUNT=0
for backup_file in "${CUDA_BACKUPS[@]}"; do
    original_file="${backup_file%.backup}"
    if [[ -f "$original_file" ]]; then
        mv "$backup_file" "$original_file"
        RESTORED_COUNT=$((RESTORED_COUNT + 1))
        echo "  ✓ Restored: $(basename "$original_file")"
    fi
done

if [[ $RESTORED_COUNT -gt 0 ]]; then
    echo "  ✓ Restored $RESTORED_COUNT CUDA files"
fi

echo ""
echo "Step 3: Verification..."

# Verify the vendor header was restored
if ! grep -q "GGML_HIP_WARP_MASK" "$VENDOR_HIP_FILE" 2>/dev/null; then
    echo "  ✓ Vendor header restoration verified"
else
    echo "  ✗ Vendor header still contains GGML_HIP_WARP_MASK - restoration may have failed"
    exit 1
fi

# Check a few CUDA files to see if they were restored
SAMPLE_CUDA_FILES=($(find "$LLAMA_DIR/ggml/src/ggml-cuda" -name "*.cu" -o -name "*.cuh" 2>/dev/null | head -3))
STILL_MODIFIED=0

for file in "${SAMPLE_CUDA_FILES[@]}"; do
    if grep -q "GGML_HIP_WARP_MASK" "$file" 2>/dev/null; then
        STILL_MODIFIED=$((STILL_MODIFIED + 1))
    fi
done

if [[ $STILL_MODIFIED -eq 0 ]]; then
    echo "  ✓ CUDA files restoration verified"
else
    echo "  ⚠ Some CUDA files may still contain GGML_HIP_WARP_MASK"
    echo "    This could be normal if not all files were originally modified"
fi

echo ""
echo "🎉 rocWMMA compatibility fixes reverted successfully!"
echo ""
echo "What was reverted:"
echo "  • Restored original ggml/src/ggml-cuda/vendors/hip.h"
echo "  • Restored original CUDA files with hardcoded warp masks"
echo ""
echo "Your llama.cpp checkout has been restored to its original state."