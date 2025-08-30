
https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html#build-from-source-rocm

install rocm and our own torch

pip install ninja cmake wheel pybind11

# or your own
pip install /opt/rocm/share/amd_smi

pip install --upgrade numba scipy huggingface-hub[cli]
pip install "numpy<2"

python use_existing_pytorch.py

pip install -r requirements/rocm-build.txt

# install our own triton again

export PYTORCH_ROCM_ARCH="gfx1151"

# Fixes needed for gfx1151 support:
# 1. Add gfx1151 to CMakeLists.txt HIP_SUPPORTED_ARCHS
# 2. Modify setup.py to handle missing torch in build isolation
# 3. Fix torch.distributed.ReduceOp import in config/parallel.py

# Use ROCm target device for setup
VLLM_TARGET_DEVICE=rocm python setup.py develop

# Alternative if that fails:
# pip install -e . --no-build-isolation
pip install "numpy<2"


wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
python benchmark_serving.py --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 64 --max-concurrency 1 --model unsloth/Llama-3.2-1B-Instruct
