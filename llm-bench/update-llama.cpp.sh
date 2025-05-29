#!/usr/bin/env bash

# Update or clone llama.cpp into various build directories and compile them.
set -euo pipefail

REPO="https://github.com/ggml-org/llama.cpp"

# Name -> cmake options
declare -A OPTS=(
  [cpu]=""
  [hip]="-DLLAMA_HIPBLAS=on"
  [rocwmma]="-DLLAMA_HIPBLAS=on -DLLAMA_USE_ROCM_WMMA=on"
  [vulkan]="-DLLAMA_VULKAN=on"
)

for name in cpu hip rocwmma vulkan; do
    dir="llama.cpp-${name}"
    if [ -d "$dir/.git" ]; then
        echo "Updating $dir"
        git -C "$dir" pull --ff-only
    else
        echo "Cloning $dir"
        git clone "$REPO" "$dir"
    fi

    cd "$dir"
    rm -rf build
    cmake -S . -B build ${OPTS[$name]}
    cmake --build build --config Release -j"$(nproc)"
    cd ..
    echo
done
