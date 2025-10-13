TheRock + AOTriton Build Guide (gfx1151)

This guide describes how to set up a Python 3.12 environment, build and install AOTriton, build PyTorch (TheRock flavor) with AOTriton enabled for Strix Halo (gfx1151), and verify Flash‑Attention works.

Prerequisites
- Conda/Mamba available
- ROCm‑capable AMD GPU (gfx1151)
- Linux x86_64

Quick Start
1) Create and activate env
   - `mamba create -n therock python=3.12 -y`
   - `mamba activate therock`

2) One‑time environment setup and ROCm SDK install
   - `./00-setup-env.sh`
   - This installs ROCm SDK via TheRock wheels and persists env vars into the `therock` env (reload on next activation).

3) Build AOTriton at the PyTorch‑pinned commit
   - `./01-build-aotriton.sh`
   - Notes:
     - Auto‑detects the AOTriton pin from `TheRock/external-builds/pytorch/pytorch/cmake/External/aotriton.cmake`.
     - Installs `pyaotriton*.so`, `libaotriton_v2.so*`, and `aotriton.images/` into your site‑packages `torch/lib/`.

4) Build PyTorch with AOTriton (gfx1151)
   - `./02-build-pytorch-with-aotriton-gfx1151.sh`
   - What it does:
     - Uses prebuilt AOTriton from step 3 if found, otherwise builds from source.
     - Applies idempotent patches to avoid CUDA‑only FBGEMM GenAI headers and links on ROCm.
     - Disables FBGEMM GenAI by default to prevent `-lfbgemm_genai` link failures.
     - Emits exact pip install commands for the wheels built for your Python (cp tag) and today’s date.
   - Tips:
     - Resume stages: `--continue pytorch` (or `triton`, `audio`, `vision`).
     - If CMake cache picks stale flags, remove only PyTorch build dir: `rm -rf TheRock/external-builds/pytorch/pytorch/build` and re‑run with `--continue pytorch`.

5) Install the built wheels
   - Use the exact commands printed by step 4. Example:
     - `pip install "/home/USER/tmp/pyout/torch-…+rocmsdkYYYYMMDD-cp312-cp312-linux_x86_64.whl" --force-reinstall --no-deps`
     - Repeat for `torchaudio` and `torchvision` (if built).

6) Validate AOTriton Flash‑Attention
   - Recommended env prefix for this session:
     - `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTORCH_ROCM_ARCH=gfx1151 TORCH_ROCM_AOTRITON_PREFER_DEFAULT=1 HIP_VISIBLE_DEVICES=0 ./run-backend-check.sh`
   - Or run directly:
     - `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTORCH_ROCM_ARCH=gfx1151 TORCH_ROCM_AOTRITON_PREFER_DEFAULT=1 HIP_VISIBLE_DEVICES=0 python backend-check.py`
   - Expected:
     - `preferred_rocm_fa_library(): _ROCmFABackend.AOTriton`
     - `can_use_flash_attention: True`
     - Forced `FLASH_ATTENTION: OK`

Environment Notes
- The setup script persists key variables for the `therock` env:
  - `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
  - `PYTORCH_ROCM_ARCH=gfx1151`
  - `TORCH_ROCM_AOTRITON_PREFER_DEFAULT=1`
  - `HIP_VISIBLE_DEVICES=0`
  - `AMD_SERIALIZE_KERNEL=1` (debug‑friendly synchronization)

Troubleshooting
- Flash/Efficient attention disabled at runtime
  - Ensure env prefix is set before importing torch (see validation step).
- `torch.ops.aotriton` shows only `['name']`
  - Normal in this packaging; SDPA still uses AOTriton. If desired, force‑load:
    - `python -c "import torch,glob,os; lib=os.path.join(os.path.dirname(torch.__file__),'lib'); c=sorted(glob.glob(os.path.join(lib,'libaotriton_v2.so*'))); [torch.ops.load_library(c[0]) if c else None]; print(dir(getattr(torch.ops,'aotriton',None)))"`
- Link error `-lfbgemm_genai`
  - The 02 script auto‑disables FBGEMM GenAI and patches sources; re‑run `--continue pytorch` after the patch.
- Old wheels in output dir
  - The 02 script prints exact wheel filepaths for this run (by cp tag + date). Use those explicit paths for pip.

References
- TheRock repository: `TheRock/`
- AOTriton pin source: `TheRock/external-builds/pytorch/pytorch/cmake/External/aotriton.cmake`
- Validation helpers: `backend-check.py`, `run-backend-check.sh`
