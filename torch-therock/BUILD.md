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
   - After it finishes: `conda deactivate && conda activate therock` to apply the persisted env vars.

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
   - Optional: sanity check direct module import
     - `python 10-test_aotriton_direct.py`

Environment Notes
- The setup script persists key variables for the `therock` env:
  - `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
  - `PYTORCH_ROCM_ARCH=gfx1151`
  - `TORCH_ROCM_AOTRITON_PREFER_DEFAULT=1`
  - `HIP_VISIBLE_DEVICES=0`
  - `AMD_SERIALIZE_KERNEL=1` (debug‑friendly synchronization)
  - Changes persisted by 00-setup-env.sh take effect on the next `conda activate therock`.

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
 - Efficient attention errors (HIP invalid argument)
   - On some ROCm nightlies for gfx11, the Efficient backend may fail. Prefer Flash‑Attention via AOTriton. For debugging: set `AMD_SERIALIZE_KERNEL=1` (or `HIP_LAUNCH_BLOCKING=1`) and retry; for performance, keep `AMD_SERIALIZE_KERNEL=0`.
 - cuDNN attention not compiled
   - Seeing “Torch was not compiled with cuDNN attention” on ROCm is normal; SDPA will select Flash/Efficient/Math instead.

References
- TheRock repository: `TheRock/`
- AOTriton pin source: `TheRock/external-builds/pytorch/pytorch/cmake/External/aotriton.cmake`
- Validation helpers: `backend-check.py`, `run-backend-check.sh`

## Composable Kernel (CK)

This section summarizes how CK relates to Flash‑Attention on ROCm, what works on gfx1151 (RDNA3), and how to benchmark fairly.

Status on gfx1151 (RDNA3)
- PyTorch’s Flash‑Attention on ROCm uses AOTriton on gfx11 (including gfx1151). This is the path you build and validate in this repo.
- CK’s fused MHA library is not emitted for gfx11 by default. CK’s build gates MHA instances to CDNA targets (gfx90a/gfx94/gfx95). See `composable_kernel/library/src/tensor_operation_instance/gpu/CMakeLists.txt:72`.
- As a result, setting `TORCH_ROCM_FA_PREFER_CK=1` on gfx1151 does not switch PyTorch SDPA to CK; `preferred_rocm_fa_library()` will still report AOTriton.

Build CK (optional: about a 1.5h build) 

- Script: `./03-build-ck.sh` (incremental by default; use `--clean` to rebuild).
  - Install prefix: defaults to `$ROCM_PATH` (if set), otherwise `$CONDA_PREFIX`, else `/opt/rocm`.
  - Recommended env‑local install: `./03-build-ck.sh --prefix "$ROCM_PATH"`
  - Targets: defaults to `gfx1151`; override with `--targets=...`.
  - Examples:
    - `./03-build-ck.sh`  # resume or configure/build for gfx1151
    - `./03-build-ck.sh --clean --prefix "$ROCM_PATH" --targets=gfx1151`
  - Under the hood: turns off `-Werror` and demotes `-Wold-style-cast` to avoid vendor header warnings breaking the build.
  - To attempt the CK MHA static lib you would need: `-DBUILD_MHA_LIB=ON` and a supported target (gfx90a/gfx94/gfx95). On gfx1151 this target gating prevents emitting `device_mha_operations`.
- Library discovery: ensure the CK install lib dir is discoverable at runtime if you experiment on supported GPUs: `export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"`.
  If installing into `/opt/rocm`, you may need `sudo` for `ninja install`. Installing into `$ROCM_PATH` avoids that and keeps changes scoped to the `therock` env.

Trying CK with PyTorch
- Selection is runtime. No PyTorch rebuild is required.
- Prefer CK (where supported): `export TORCH_ROCM_FA_PREFER_CK=1`.
- Verify selection: `python backend-check.py` prints `preferred_rocm_fa_library()`. On gfx1151 it should remain `_ROCmFABackend.AOTriton`.
- For AOTriton, keep: `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTORCH_ROCM_ARCH=gfx1151`.

Benchmarking and AMD_SERIALIZE_KERNEL
- `AMD_SERIALIZE_KERNEL` controls extra synchronization to aid debugging:
  - `AMD_SERIALIZE_KERNEL=1`: forces device‑wide synchronization between kernel launches. Safer for tracing/debugging; significantly hurts throughput and can mask overlap.
  - `AMD_SERIALIZE_KERNEL=0`: no extra sync beyond what kernels require. Use this for normal usage and performance benchmarking.
- Recommended for timing: `AMD_SERIALIZE_KERNEL=0`.

Quick commands
- Baseline AOTriton benchmark on gfx1151:
  - `AMD_SERIALIZE_KERNEL=0 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTORCH_ROCM_ARCH=gfx1151 HIP_VISIBLE_DEVICES=0 python 05-attention-bench.py`
- Attempt CK preference + verify (will still choose AOTriton on gfx1151):
  - `AMD_SERIALIZE_KERNEL=0 TORCH_ROCM_FA_PREFER_CK=1 HIP_VISIBLE_DEVICES=0 python backend-check.py`

Notes
- CK’s WMMA attention examples can be built for gfx11, but the fused MHA library that PyTorch’s SDPA expects is currently gated to CDNA targets. Integrating CK MHA on gfx11 would require upstream changes beyond environment flags.
- You can flip between AOTriton and “prefer CK” runs by environment variables; no PyTorch rebuilds are necessary just to test selection.
