# HOWTO: Install TheRock ROCm + PyTorch (gfx1151)

This guide sets up a working PyTorch environment using AMD’s TheRock nightly wheels for Strix Halo (gfx1151), including Flash‑Attention via [ao]triton (AOTriton) on Linux by pinning torch < 2.10.

NOTE: as of 2025-10 NO PREPACKAGED VERSION OF TheRock PyTorch for gfx1151 has AOTriton (and hence, Flash Attention) support.

## Prerequisites

- AMD Strix Halo iGPU (gfx1151)
- Conda/Mamba (recommended) and Python 3.12

## Install Steps

```bash
mamba create -n therock-default python=3.12 -y
mamba activate therock-default

# NOTE: Install developer tools/headers BEFORE torch to avoid resolver backtracking
# and keep ROCm packages aligned during the initial solve. You can also pin after
# installing torch (see commands below).
python -m pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[devel]"

# PyTorch + ROCm for gfx1151 with AOTriton enabled (Linux requires torch < 2.10)
python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  --pre "torch<2.10" torchaudio torchvision
```

### Keep rocm[devel] matching torch’s ROCm version

If you installed `rocm[devel]` earlier (or want to add it after installing torch),
pin it to the exact ROCm version that torch resolved, using the `rocm-sdk version`
CLI. This guarantees the dev tools/headers match the runtime already pulled in by torch.

- Bash:

```bash
python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  "rocm[devel]==$(rocm-sdk version)"
```

- Fish:

```fish
set v (rocm-sdk version)
python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  "rocm[devel]==$v"
```

Notes:
- The `torch` wheel automatically depends on `rocm[libraries]` and `pytorch-triton-rocm`, so the ROCm runtime is pulled in automatically (no separate ROCm install needed for runtime).
- The gfx1151 index and install incantation are documented in TheRock releases: `TheRock/RELEASES.md:321`.

## Verify

```bash
rocm-sdk test  # quick sanity check of the ROCm Python packages

python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('Has aotriton:', hasattr(torch.ops, 'aotriton'))
PY
```

Expected:
- `rocm-sdk test` passes.
- `torch.cuda.is_available()` is True, device is your AMD GPU.
- `hasattr(torch.ops, 'aotriton')` is True for torch < 2.10 on Linux.

## Why pin torch < 2.10 on Linux for AOTriton

- TheRock’s build docs state: Flash attention via [ao]triton is enabled on Linux for torch < 2.10 and disabled for torch ≥ 2.10 (pending fixes).
  - See `TheRock/external-builds/pytorch/README.md:33`.
- The TheRock build script enforces this behavior on Linux (sets `USE_FLASH_ATTENTION=1` for PyTorch < 2.10, otherwise 0 unless explicitly overridden when building from source).
  - See `TheRock/external-builds/pytorch/build_prod_wheels.py:625`.

If you installed 2.10+ and need FA now, uninstall and reinstall with the constraint:

```bash
python -m pip uninstall -y torch torchvision torchaudio pytorch-triton-rocm
python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  --pre "torch>=2.9,<2.10" "torchaudio>=2.9,<2.10" "torchvision>=0.24,<0.25"
```

## Optional: Exact version pin

TheRock’s releases show an example pin you can use to keep FA working:

```bash
python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  --pre "torch==2.7.1" "torchaudio==2.7.1a0" "torchvision==0.22.1"
```

Reference: `TheRock/RELEASES.md:268`.

## Additional notes

- You do not need to set conda env vars or `LD_LIBRARY_PATH` for runtime. The wheels include an initializer (`_rocm_init.py`) that calls `rocm_sdk.initialize_process()` to preload required ROCm libraries automatically.
  - See `TheRock/docs/packaging/python_packaging.md:145` and `TheRock/external-builds/pytorch/build_prod_wheels.py:228`.
- If you plan to compile against ROCm tools/headers for other native projects, keep `rocm[devel]` installed. Paths can be queried via:
  - `python -m rocm_sdk path --root`
  - `python -m rocm_sdk path --bin`
