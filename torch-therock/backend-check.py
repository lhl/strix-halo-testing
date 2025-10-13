#!/usr/bin/env python3

import os
import sys
import glob


def _sanitize_env() -> None:
    # Avoid noisy warnings: AMD_SERIALIZE_KERNEL is treated as boolean in PyTorch env parsing
    v = os.environ.get("AMD_SERIALIZE_KERNEL")
    if v and v.strip() not in ("0", "1"):
        # Use 1 to keep debugging-friendly sync behavior if user passed a non-boolean value
        os.environ["AMD_SERIALIZE_KERNEL"] = "1"


def _maybe_force_load_aotriton(torch) -> None:
    try:
        libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
        cands = sorted(glob.glob(os.path.join(libdir, "libaotriton_v2.so*")))
        if cands:
            torch.ops.load_library(cands[0])
    except Exception:
        # Best effort only; if this fails, SDPA can still run via AOTriton
        pass


def main() -> int:
    _sanitize_env()
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        print(f"Failed to import torch: {e}")
        return 1

    # Prefer the new API surface for SDPA
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from torch.backends.cuda import (
        is_flash_attention_available,
        flash_sdp_enabled,
        mem_efficient_sdp_enabled,
        math_sdp_enabled,
        preferred_rocm_fa_library,
        SDPAParams,
        can_use_flash_attention,
        can_use_efficient_attention,
    )

    print("=== SDPA Backend Check ===")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {getattr(torch.version, 'hip', None)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"Device 0: {torch.cuda.get_device_name(0)}")
            print(f"Capability: {torch.cuda.get_device_capability(0)}")
        except Exception as e:
            print(f"Device query error: {e}")

    # Environment overview
    print("\n=== Environment Variables ===")
    for var in (
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
        "PYTORCH_ROCM_ARCH",
        "TORCH_ROCM_FA_PREFER_CK",
        "HSA_OVERRIDE_GFX_VERSION",
    ):
        print(f"{var}: {os.getenv(var, 'Not set')}")

    # AOTriton ops visibility
    _maybe_force_load_aotriton(torch)
    print("\n=== AOTriton Ops ===")
    ops = getattr(torch.ops, "aotriton", None)
    print("torch.ops.aotriton:", [x for x in dir(ops) if not x.startswith("_")] if ops else "none")

    # SDPA feature flags
    print("\n=== SDPA Feature Flags ===")
    print(f"is_flash_attention_available: {is_flash_attention_available()}")
    print(f"flash_sdp_enabled: {flash_sdp_enabled()}")
    print(f"mem_efficient_sdp_enabled: {mem_efficient_sdp_enabled()}")
    print(f"math_sdp_enabled: {math_sdp_enabled()}")
    try:
        from torch.backends.cuda import cudnn_sdp_enabled  # type: ignore[attr-defined]
        print(f"cudnn_sdp_enabled: {cudnn_sdp_enabled()}")
    except Exception:
        print("cudnn_sdp_enabled: n/a")
    print(f"preferred_rocm_fa_library(): {preferred_rocm_fa_library()}")

    if not torch.cuda.is_available():
        print("\nCUDA/HIP device not available; skipping runtime backend execution checks.")
        return 0

    # Prepare tiny tensors
    device = "cuda"
    dtype = torch.float16
    q = torch.randn(1, 1, 128, 64, device=device, dtype=dtype)
    k = torch.randn(1, 1, 128, 64, device=device, dtype=dtype)
    v = torch.randn(1, 1, 128, 64, device=device, dtype=dtype)

    # Check whether specific optimized backends can be used for these params
    print("\n=== Backend Viability (can_use_*) ===")
    try:
        params = SDPAParams(q, k, v, None, 0.0, True, False)
        print("can_use_flash_attention:", can_use_flash_attention(params, debug=True))
        print("can_use_efficient_attention:", can_use_efficient_attention(params, debug=True))
    except TypeError as e:
        print(f"SDPAParams construction failed (signature mismatch?): {e}")

    # Try executing with each backend explicitly
    print("\n=== Forced Execution Per Backend ===")
    def try_backend(backend):
        try:
            with sdpa_kernel(backend):
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return "OK"
        except Exception as ex:
            return f"ERR: {ex}"

    checks = [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
        ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
    ]
    for name, be in checks:
        msg = try_backend(be)
        print(f"{name}: {msg}")
        if name == "EFFICIENT_ATTENTION" and msg.startswith("ERR"):
            print("Note: Efficient attention can be unstable on current ROCm nightlies. Prefer FLASH_ATTENTION.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
