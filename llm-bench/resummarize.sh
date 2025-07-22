#!/bin/bash

# resummarize.sh - Easily resummarize all model benchmark results
# This script finds all directories containing results.jsonl and regenerates their README.md

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHER="$SCRIPT_DIR/llama-cpp-bencher.py"

if [[ ! -f "$BENCHER" ]]; then
    echo "Error: llama-cpp-bencher.py not found at $BENCHER"
    exit 1
fi

# Find all directories containing results.jsonl
model_dirs=()
while IFS= read -r -d '' dir; do
    model_dirs+=("$(dirname "$dir")")
done < <(find "$SCRIPT_DIR" -name "results.jsonl" -print0)

if [[ ${#model_dirs[@]} -eq 0 ]]; then
    echo "No model directories with results.jsonl found"
    exit 1
fi

echo "Found ${#model_dirs[@]} model directories to resummarize:"
printf '%s\n' "${model_dirs[@]}" | sort

echo ""
echo "Resummarizing..."

for model_dir in "${model_dirs[@]}"; do
    model_name="$(basename "$model_dir")"
    echo "Processing $model_name..."
    
    if ! python3 "$BENCHER" --outdir "$model_dir" --resummarize; then
        echo "Warning: Failed to resummarize $model_name"
        continue
    fi
    
    echo "✓ Updated README.md for $model_name"
done

echo ""
echo "All models resummarized successfully!"