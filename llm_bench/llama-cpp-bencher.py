#!/usr/bin/env python3
"""llama_cpp_bencher.py

Benchmark multiple **llama.cpp** builds across prompt‑processing (PP) and
token‑generation (TG) sweeps, capture peak memory, and emit:

* **results.jsonl** – one record per run
* **PNG charts** per build for PP & TG sweeps
* **summary_<model>.md** – quick Markdown table (PP512 & TG128)

### Key changes (2025‑05‑28)
* **Default build roots** now under **/home/lhl/llama.cpp-* **.
* Sweep logic matches upstream *llama-bench* semantics:
  * **PP sweep:** `-p <pp>`  & `-n 0`
  * **TG sweep:** `-p 0`     & `-n <tg>`
* `--outdir` is **optional**.  If omitted, the model’s stem (e.g. *llama-2-7b.Q4_0*)
  becomes the output directory.

Run example:
```bash
python llama_cpp_bencher.py \
  --model /models/gguf/llama-2-7b.Q4_0.gguf \
  --flags "-fa 1 -mmap"
```
This creates `llama-2-7b.Q4_0/` alongside your working dir, containing all
artifacts.
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
import time
import textwrap
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

###############################################################################
# ------------------------------- CONFIG ------------------------------------ #
###############################################################################

BUILD_ROOT = Path("/home/lhl/llama.cpp")  # where llama.cpp-* folders live

DEFAULT_BUILDS = {
    name: str(BUILD_ROOT / f"{name}/build/bin/llama-bench")
    for name in [
        "llama.cpp-cpu",
        "llama.cpp-hip",
        "llama.cpp-hipblaslt",
        "llama.cpp-hip-uma",
        "llama.cpp-hjc4869",
        "llama.cpp-rocwmma",
        "llama.cpp-vulkan",
    ]
}

# Prompt‑token and generation‑token sweeps (powers of two 1→8192)
DEFAULT_PP_VALUES = [2 ** i for i in range(0, 14)]
DEFAULT_TG_VALUES = DEFAULT_PP_VALUES.copy()

DEFAULT_FLAGS = "-fa 1"  # extra llama‑bench flags

###############################################################################
# ---------------------------- MEMORY MONITORS ------------------------------ #
###############################################################################

def _first_int(text: str) -> int:
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else 0


def monitor_rocm_vram(stop: Event, peaks: Dict[str, int], interval: float = 1.0):
    initial = peak = 0
    while not stop.is_set():
        try:
            out = sp.check_output(["rocm-smi", "--showmeminfo", "vram", "--csv"], text=True)
            used_bytes = int(out.splitlines()[1].split(",")[2])
            used = used_bytes // (1024 * 1024)
            if not initial:
                initial = used
            peak = max(peak, used)
        except Exception:
            pass
        time.sleep(interval)
    if peak:
        peaks["vram_peak_mib"] = peak


def monitor_gtt(stop: Event, peaks: Dict[str, int], interval: float = 1.0):
    initial = peak = 0
    while not stop.is_set():
        try:
            out = sp.check_output(["amdgpu_top", "-d"], text=True)
            line = next((l for l in out.splitlines() if re.match(r"^\s*GTT", l)), "")
            used = _first_int(line)
            if not initial:
                initial = used
            peak = max(peak, used)
        except Exception:
            pass
        time.sleep(interval)
    if peak:
        peaks["gtt_peak_mib"] = peak


def monitor_sysram(stop: Event, peaks: Dict[str, int], interval: float = 1.0):
    initial = peak = 0
    while not stop.is_set():
        try:
            with open("/proc/meminfo") as fh:
                memtotal = memavail = 0
                for l in fh:
                    if l.startswith("MemTotal"):
                        memtotal = _first_int(l)
                    elif l.startswith("MemAvailable"):
                        memavail = _first_int(l)
            used = (memtotal - memavail) // 1024  # kB → MiB
            if not initial:
                initial = used
            peak = max(peak, used)
        except Exception:
            pass
        time.sleep(interval)
    if peak:
        peaks["system_ram_peak_mib"] = peak

###############################################################################
# ------------------------- BENCHMARK EXECUTION ----------------------------- #
###############################################################################

def run_bench(binary: str, model: str, flags: str, mode: str, value: int, gpu_mon: bool) -> Dict:
    """Run one llama‑bench invocation and return collected metrics."""

    if mode == "pp":
        bench_args = f"-p {value} -n 0"
    elif mode == "tg":
        bench_args = f"-p 0 -n {value}"
    else:
        raise ValueError("mode must be 'pp' or 'tg'")

    cmd = f"{shlex.quote(binary)} -m {shlex.quote(model)} {bench_args} {flags}"
    print("\n[RUN]", cmd)

    stop = Event()
    peaks: Dict[str, int] = {}
    monitors = [Thread(target=monitor_sysram, args=(stop, peaks), daemon=True)]
    if gpu_mon:
        monitors += [
            Thread(target=monitor_rocm_vram, args=(stop, peaks), daemon=True),
            Thread(target=monitor_gtt, args=(stop, peaks), daemon=True),
        ]
    for t in monitors:
        t.start()

    tok_s = 0.0
    ttft_ms = None
    start = time.time()

    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            try:
                rec = json.loads(line)
                tok_s = rec.get("tok_per_s", tok_s)
                ttft_ms = rec.get("ttft_ms", ttft_ms)
            except Exception:
                pass
    finally:
        proc.wait()
        stop.set()
        for t in monitors:
            t.join()

    rec = {
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "binary": binary,
        "mode": mode,
        "value": value,
        "tokens_per_sec": tok_s,
        "ttft_ms": ttft_ms,
        "runtime_s": time.time() - start,
    }
    rec.update(peaks)
    return rec

###############################################################################
# -------------------------- PLOTTING & TABLES ------------------------------ #
###############################################################################

def plot_series(df: pd.DataFrame, build: str, mode: str, outdir: Path):
    sub = df[(df.build == build) & (df.mode == mode)].sort_values("value")
    if sub.empty:
        return
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_title(f"{build} – {'PP' if mode=='pp' else 'TG'} sweep")
    ax1.set_xlabel("Token count")
    ax1.set_ylabel("tokens/s")
    ax1.plot(sub.value, sub.tokens_per_sec, marker="o")

    ax2 = ax1.twinx()
    ax2.set_ylabel("VRAM peak (MiB)")
    ax2.plot(sub.value, sub.vram_peak_mib, linestyle="--", marker="x")

    ax1.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(outdir / f"{build}_{mode}.png", dpi=150)
    plt.close(fig)


def write_summary(df: pd.DataFrame, model: str, outdir: Path):
    core_cols = ["build", "tokens_per_sec", "vram_peak_mib", "gtt_peak_mib", "system_ram_peak_mib", "ttft_ms"]
    rows = []
    for build in df.build.unique():
        pp = df[(df.build == build) & (df.mode == "pp") & (df.value == 512)].iloc[0:1]
        tg = df[(df.build == build) & (df.mode == "tg") & (df.value == 128)].iloc[0:1]
        rows.extend([pp, tg])
    md = tabulate(pd.concat(rows)[core_cols], headers="keys", tablefmt="github")
    (outdir / f"summary_{model}.md").write_text(md)

###############################################################################
# --------------------------------- MAIN ------------------------------------ #
###############################################################################

def main():
    p = argparse.ArgumentParser(description=textwrap.dedent(__doc__), formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to GGUF model")
    p.add_argument("--outdir", help="Directory for outputs (defaults to model stem)")
    p.add_argument("--builds", nargs="*", help="Subset of build names or explicit binaries")
    p.add_argument("--flags", default=DEFAULT_FLAGS, help="Extra flags passed verbatim to llama-bench")
    p.add_argument("--pp", nargs="*", type=int, help="Prompt sizes to sweep")
    p.add_argument("--tg", nargs="*", type=int, help="Generation sizes to sweep")
    p.add_argument("--skip_gpu_mon", action="store_true", help="Disable GPU memory monitors")
    args = p.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(Path(args.model).stem)
    outdir.mkdir(parents=True, exist_ok=True)

    builds = args.builds or list(DEFAULT_BUILDS.keys())
    pp_vals = args.pp or DEFAULT_PP_VALUES
    tg_vals = args.tg or DEFAULT_TG_VALUES

    jsonl_path = outdir / "results.jsonl"
    with open(jsonl_path, "w") as jl:
        for build in builds:
            binary = DEFAULT_BUILDS.get(build, build)  # explicit path passes through
            commit = ""
            repo = Path(binary).parent.parent
            if (repo / ".git").exists():
                try:
                    commit = sp.check_output(["git", "--git-dir", str(repo / ".git"), "rev-parse", "--short", "HEAD"], text=True).strip()
                except Exception:
                    pass
            for mode, sweep in (("pp", pp_vals), ("tg", tg_vals)):
                for val in sweep:
                    rec = run_bench(binary, args.model, args.flags, mode, val, not args.skip_gpu_mon)
                    rec.update({"build": build, "commit": commit})
                    jl.write(json.dumps(rec) + "\n")
                    jl.flush()

    df = pd.read_json(jsonl_path, lines=True)
    for build in df.build.unique():
        for mode in ["pp", "tg"]:
            plot_series(df, build, mode, outdir)
    write_summary(df, Path(args.model).stem, outdir)
    print("\nFinished!  Artifacts in", outdir)


if __name__ == "__main__":
    main()

