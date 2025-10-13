#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper to run backend-check.py with AOTriton + gfx1151 env
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}
export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx1151}
export TORCH_ROCM_AOTRITON_PREFER_DEFAULT=${TORCH_ROCM_AOTRITON_PREFER_DEFAULT:-1}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

python backend-check.py "$@"

