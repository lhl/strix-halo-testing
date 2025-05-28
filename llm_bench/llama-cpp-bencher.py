#!/usr/bin/env python3
"""llama_bench_suite.py

Benchmark multiple llama.cpp builds across prompt‑processing (PP) and
token‑generation (TG) sweeps, capture peak memory, and produce summary
artifacts (JSONL, Markdown, PNG charts).

Usage (minimal):
    python llama_bench_suite.py \
        --model /models/gguf/shisa-v2-llama3.3-70b.i1-Q4_K_M.gguf \
        --outdir results_70b

Many parameters can be overridden – see `--help`.

The script is Linux‑only and expects AMD GPUs with ROCm for GPU backends
and the following helper tools available on $PATH where relevant:
    * rocm-smi           – VRAM usage (GPU backends)
    * amdgpu_top -d      – GTT usage (GPU backends)
    * free /proc/meminfo – system RAM (CPU backend)

JSONL schema (one record per run):
{
  "timestamp": "2025-05-28T06:42:12+09:00",
  "build": "llama.cpp-hip",
  "binary": "/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench",
  "flags": "-fa 1 -mmap",
  "mode": "pp",                   # or "tg"
  "value": 512,                   # prompt‑tokens or gen‑tokens
  "tokens_per_sec": 335.9,
  "ttft_ms": 123.4,               # time‑to‑first‑token (if derivable)
  "vram_peak_mib": 24312,
  "gtt_peak_mib": 1987,
  "system_ram_peak_mib": 31294,
  "runtime_s": 87.2,
  "kernel": "6.9.2-arch1-1",
  "rocm_version": "6.1.1",
  "commit": "1a2b3c4 (2025-05-10)"
}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shlex
import subprocess as sp
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

###############################################################################
# ------------------------------- CONFIG ------------------------------------ #
###############################################################################

# Default llama.cpp builds to test (override with --builds)
DEFAULT_BUILDS = {
    "llama.cpp-cpu": "/home/lhl/llama.cpp/llama.cpp-cpu/build/bin/llama-bench",
    "llama.cpp-hip": "/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench",
    "llama.cpp-hipblaslt": "/home/lhl/llama.cpp/llama.cpp-hipblaslt/build/bin/llama-bench",
    "llama.cpp-hip-uma": "/home/lhl/llama.cpp/llama.cpp-hip-uma/build/bin/llama-bench",
    "llama.cpp-hjc4869": "/home/lhl/llama.cpp/llama.cpp-hjc4869/build/bin/llama-bench",
    "llama.cpp-rocwmma": "/home/lhl/llama.cpp/llama.cpp-rocwmma/build/bin/llama-bench",
    "llama.cpp-vulkan": "/home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-bench",
}

# Prompt‑processing token counts (powers of two 1→8192)
DEFAULT_PP_VALUES = [2 ** i for i in range(0, 14)]  # 1 … 8192
# Generation token counts (same list by default)
DEFAULT_TG_VALUES = DEFAULT_PP_VALUES.copy()

# Additional static flags to pass to llama-bench (may be overridden via CLI)
DEFAULT_FLAGS = "-fa 1"

###############################################################################
# ---------------------------- MEMORY MONITORS ------------------------------ #
###############################################################################

def _parse_first_int(text: str) -> int:
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else 0


def monitor_rocm_vram(interval: float, stop: Event, result: Dict[str, int]):
    """Monitors VRAM via rocm-smi (MiB) and stores peak."""
    initial = peak = 0
    while not stop.is_set():
        try:
            out = sp.check_output(["rocm-smi", "--showmeminfo", "vram", "--csv"], text=True)
            # second CSV line, 3rd column = used bytes
            used_bytes = int(out.splitlines()[1].split(",")[2])
            used = used_bytes // (1024 * 1024)
            if initial == 0:
                initial = used
            if used > peak:
                peak = used
        except Exception:
            pass
        time.sleep(interval)
    if peak:
        result["vram_peak_mib"] = peak


def monitor_gtt(interval: float, stop: Event, result: Dict[str, int]):
    """Monitors GTT via amdgpu_top -d (MiB)."""
    initial = peak = 0
    while not stop.is_set():
        try:
            out = sp.check_output(["amdgpu_top", "-d"], text=True)
            gtt_line = next((l for l in out.splitlines() if re.search(r"^\s*GTT", l)), "")
            used = _parse_first_int(gtt_line)
            if initial == 0:
                initial = used
            if used > peak:
                peak = used
        except Exception:
            pass
        time.sleep(interval)
    if peak:
        result["gtt_peak_mib"] = peak


def monitor_sysram(interval: float, stop: Event, result: Dict[str, int]):
    """Monitors system RAM (MiB) via /proc/meminfo."""
    initial = peak = 0
    while not stop.is_set():
        try:
            with open("/proc/meminfo") as fh:
                memfree = 0
                memtotal = 0
                for line in fh:
                    if line.startswith("MemTotal"):
                        memtotal = _parse_first_int(line)
                    if line.startswith("MemAvailable"):
                        memfree = _parse_first_int(line)
                used = (memtotal - memfree) // 1024  # kB→MiB
                if initial == 0:
                    initial = used
                if used > peak:
                    peak = used
        except Exception:
            pass
        time.sleep(interval)
    if peak:
        result["system_ram_peak_mib"] = peak

###############################################################################
# ------------------------- BENCHMARK EXECUTION ----------------------------- #
###############################################################################

def run_single_bench(binary: str, model: str, flags: str, mode: str, value: int,
                      monitor_gpu: bool = True) -> Dict:
    """Run one llama-bench invocation and capture metrics and memory peaks."""

    cmd = f"{shlex.quote(binary)} -m {shlex.quote(model)} -n 0 {flags}"
    if mode == "pp":
        cmd += f" -p {value}"
    elif mode == "tg":
        cmd += f" -n {value}"
    else:
        raise ValueError("mode must be 'pp' or 'tg'")

    print("\n[RUN]", cmd)
    stop_evt = Event()
    peaks: Dict[str, int] = {}

    monitors: List[Thread] = []
    if monitor_gpu:
        monitors.extend([
            Thread(target=monitor_rocm_vram, args=(1.0, stop_evt, peaks), daemon=True),
            Thread(target=monitor_gtt, args=(1.0, stop_evt, peaks), daemon=True),
        ])
    monitors.append(Thread(target=monitor_sysram, args=(1.0, stop_evt, peaks), daemon=True))
    for t in monitors:
        t.start()

    # Capture stdout
    start_t = time.time()
    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1)
    tokens_per_sec = 0.0
    ttft_ms = None
    try:
        for line in proc.stdout:
            # Quickly parse JSONL lines produced by llama-bench for t/s etc.
            try:
                rec = json.loads(line)
                if "tok_per_s" in rec:
                    tokens_per_sec = rec["tok_per_s"]
                if "ttft_ms" in rec:
                    ttft_ms = rec["ttft_ms"]
            except Exception:
                pass  # non‑JSON log line
            sys.stdout.write(line)  # stream through
    finally:
        proc.wait()
        runtime_s = time.time() - start_t
        stop_evt.set()
        for t in monitors:
            t.join()

    result = {
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "binary": binary,
        "flags": flags,
        "mode": mode,
        "value": value,
        "tokens_per_sec": tokens_per_sec,
        "ttft_ms": ttft_ms,
        "runtime_s": runtime_s,
    }
    result.update(peaks)
    return result

###############################################################################
# -------------------------- PLOTTING & TABLES ------------------------------ #
###############################################################################

def plot_series(df: pd.DataFrame, build: str, mode: str, outdir: Path):
    """Plot tokens/s and memory peak over value sweep for one build."""

    fig, ax1 = plt.subplots(figsize=(8, 5))
    subset = df[df["build"] == build]
    subset = subset[subset["mode"] == mode].sort_values("value")

    ax1.set_title(f"{build} – {'PP' if mode=='pp' else 'TG'} sweep")
    ax1.set_xlabel("Token count")
    ax1.set_ylabel("tokens/s")
    ax1.plot(subset["value"], subset["tokens_per_sec"], marker="o", label="tokens/s")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Peak VRAM (MiB)")
    ax2.plot(subset["value"], subset["vram_peak_mib"], linestyle="--", marker="x", label="VRAM peak")
    ax1.grid(True, which="both", alpha=0.3, linestyle=":")

    fig.tight_layout()
    out_path = outdir / f"{build}_{mode}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved", out_path)


def write_markdown_table(df: pd.DataFrame, model: str, outdir: Path):
    pp_row = df[(df.mode == "pp") & (df.value == 512)].copy()
    tg_row = df[(df.mode == "tg") & (df.value == 128)].copy()
    core_cols = [
        "build", "tokens_per_sec", "vram_peak_mib", "gtt_peak_mib", "system_ram_peak_mib", "ttft_ms"
    ]
    md = "### Summary @PP512 / TG128\n\n" + tabulate(pd.concat([pp_row, tg_row])[core_cols], headers="keys", tablefmt="github")
    with open(outdir / f"summary_{model}.md", "w") as fh:
        fh.write(md)

###############################################################################
# --------------------------------- MAIN ------------------------------------ #
###############################################################################

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__))
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--outdir", required=True, help="Directory to save results")
    parser.add_argument("--builds", nargs="*", help="Space‑separated list of build names to test")
    parser.add_argument("--flags", default=DEFAULT_FLAGS, help="Extra flags to pass to llama-bench")
    parser.add_argument("--pp", nargs="*", type=int, help="Prompt token counts")
    parser.add_argument("--tg", nargs="*", type=int, help="Generation token counts")
    parser.add_argument("--skip_gpu_mon", action="store_true", help="Disable GPU memory monitors")
    args = parser.parse_args()

    builds = args.builds or list(DEFAULT_BUILDS.keys())
    pp_vals = args.pp or DEFAULT_PP_VALUES
    tg_vals = args.tg or DEFAULT_TG_VALUES
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jsonl_path = outdir / "results.jsonl"
    with open(jsonl_path, "w") as jl:
        for build in builds:
            binary = DEFAULT_BUILDS.get(build) or build  # allow passing explicit path
            # commit id if git repo present
            commit = ""
            repo_dir = Path(binary).parent.parent
            head_path = repo_dir / ".git/HEAD"
            if head_path.exists():
                commit = sp.check_output(["git", "--git-dir", str(repo_dir/".git"), "rev-parse", "--short", "HEAD"], text=True).strip()
            for mode, sweep in [("pp", pp_vals), ("tg", tg_vals)]:
                for val in sweep:
                    rec = run_single_bench(
                        binary=binary,
                        model=args.model,
                        flags=args.flags,
                        mode=mode,
                        value=val,
                        monitor_gpu=not args.skip_gpu_mon,
                    )
                    rec.update({"build": build, "commit": commit})
                    jl.write(json.dumps(rec) + "\n")
                    jl.flush()

    # ----------- analysis & plots ------------- #
    df = pd.read_json(jsonl_path, lines=True)
    for build in df.build.unique():
        for mode in ["pp", "tg"]:
            plot_series(df, build, mode, outdir)
    write_markdown_table(df, Path(args.model).stem, outdir)

    print("\nAll done! Results in", outdir)


if __name__ == "__main__":
    main()

