#!/usr/bin/env python3
"""AMD LLM benchmarking utility.

This script runs `llama-bench` with various configurations and collects
performance metrics such as memory usage, power and temperature. Results are
stored in a timestamped directory containing raw command output and a summary
JSON file.  A configuration JSON file defines which commands to run.

The script attempts to generate simple graphs using matplotlib if it is
available.  If matplotlib is missing the graph generation step is skipped.
"""

import argparse
import json
import os
import platform
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def run_cmd(cmd: str) -> str:
    """Run a shell command and return its stdout as a string."""
    try:
        out = subprocess.check_output(cmd, shell=True, text=True)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()


def gather_system_info(executable: str) -> Dict[str, str]:
    """Collect detailed system and software version information."""
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hostname": platform.node(),
        "kernel": run_cmd("uname -r"),
        "os": platform.platform(),
        "cpu": run_cmd("lscpu | grep 'Model name' | awk -F: '{print $2}'"),
        "rocm_version": run_cmd("rocminfo | grep -m1 'Runtime Version' | awk '{print $3}' || true"),
        "hip_version": run_cmd("hipcc --version 2>/dev/null | head -n1 || true"),
        "vulkan_version": run_cmd("vulkaninfo --summary 2>/dev/null | awk -F: '/Vulkan Instance Version/{print $2;exit}' || true"),
        "amdgpu_driver": run_cmd("modinfo -F version amdgpu 2>/dev/null || true"),
        "inxi": run_cmd("inxi -b 2>/dev/null || true"),
    }

    # attempt to capture the llama.cpp commit used by the executable
    try:
        exe_path = Path(executable).resolve()
        repo_root = exe_path.parents[2]
        commit = run_cmd(f"git -C {repo_root} rev-parse HEAD 2>/dev/null || true")
        if commit:
            info["llama_cpp_commit"] = commit
    except Exception:
        pass

    gpu_name = run_cmd("rocm-smi --showproductname --json || true")
    if gpu_name:
        info["gpu"] = gpu_name

    # collect additional GPU details from amdgpu_top if available
    amd_top = run_cmd("amdgpu_top -d 2>/dev/null || true")
    if amd_top:
        import re

        def search(pattern: str) -> Optional[str]:
            m = re.search(pattern, amd_top)
            return m.group(1).strip() if m else None

        gpu_details = {
            "device_name": search(r"device_name:\s*\"([^\"]+)\""),
            "device_type": search(r"device_type:\s*([A-Za-z0-9_]+)"),
            "gpu_type": search(r"GPU Type\s*:\s*([^\n]+)"),
            "family": search(r"Family\s*:\s*([^\n]+)"),
            "asic_name": search(r"ASIC Name\s*:\s*([^\n]+)"),
            "chip_class": search(r"Chip Class\s*:\s*([^\n]+)"),
        }

        # parse VBIOS info block
        vbios_block = re.search(r"VBIOS info:\n(.*?)(?:\n\s*\n|$)", amd_top, re.S)
        if vbios_block:
            vbios = {}
            for line in vbios_block.group(1).splitlines():
                parts = [p.strip(" []") for p in line.split(":", 1)]
                if len(parts) == 2:
                    vbios[parts[0]] = parts[1]
            gpu_details["vbios"] = vbios

        # parse Firmware info block
        fw_block = re.search(r"Firmware info:\n(.*?)(?:\n\s*\n|$)", amd_top, re.S)
        if fw_block:
            fw_info = {}
            for line in fw_block.group(1).splitlines():
                m = re.match(r"\s*(\S+)\s+feature:\s*(\d+),\s+ver:\s*(\S+)", line)
                if m:
                    fw_info[m.group(1)] = {"feature": int(m.group(2)), "ver": m.group(3)}
            gpu_details["firmware"] = fw_info

        info["gpu_details"] = gpu_details
    return info


class MetricWatcher(threading.Thread):
    """Background watcher for numeric metrics."""

    def __init__(self, name: str, cmd: str, interval: float = 1.0) -> None:
        super().__init__()
        self.name = name
        self.cmd = cmd
        self.interval = interval
        self.initial: Optional[int] = None
        self.max_value: Optional[int] = None
        self._stop = threading.Event()

    def _get(self) -> Optional[int]:
        out = run_cmd(self.cmd)
        try:
            return int(out.split()[0])
        except (ValueError, IndexError):
            return None

    def run(self) -> None:
        self.initial = self._get()
        self.max_value = self.initial
        while not self._stop.is_set():
            time.sleep(self.interval)
            val = self._get()
            if val is not None and self.max_value is not None:
                if val > self.max_value:
                    self.max_value = val

    def stop(self) -> None:
        self._stop.set()


class SensorWatcher(threading.Thread):
    """Track maximum edge temperature and power using the `sensors` command."""

    def __init__(self, interval: float = 1.0) -> None:
        super().__init__()
        self.interval = interval
        self.max_temp: Optional[float] = None
        self.max_power: Optional[float] = None
        self._power_samples: List[float] = []
        self._stop = threading.Event()

    def _poll(self) -> None:
        out = run_cmd("sensors")
        for line in out.splitlines():
            if "edge:" in line:
                try:
                    val = float(line.split()[1].strip("+°C"))
                    if self.max_temp is None or val > self.max_temp:
                        self.max_temp = val
                except ValueError:
                    pass
            if "PPT:" in line:
                try:
                    val = float(line.split()[1])
                    if self.max_power is None or val > self.max_power:
                        self.max_power = val
                    self._power_samples.append(val)
                except ValueError:
                    pass

    def run(self) -> None:
        while not self._stop.is_set():
            self._poll()
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()

    def avg_power(self) -> Optional[float]:
        if not self._power_samples:
            return None
        return sum(self._power_samples) / len(self._power_samples)

    def median_power(self) -> Optional[float]:
        if not self._power_samples:
            return None
        s = sorted(self._power_samples)
        mid = len(s) // 2
        if len(s) % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2
        return s[mid]


# ----------------------------------------------------------------------------
# main benchmarking logic
# ----------------------------------------------------------------------------


def run_benchmark(cmd: List[str], log_dir: Path) -> List[Dict]:
    """Run a single llama-bench command and capture its JSONL output."""
    log_file = log_dir / "stdout.txt"
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc, log_file.open("w") as f:
        lines = []
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            lines.append(line)
        proc.wait()
    results = []
    for line in lines:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def generate_graph(results: Dict[str, List[Dict]], out_dir: Path) -> None:
    """Generate simple line graphs if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available - skipping graph generation")
        return

    pp_x, pp_y = [], []
    tg_x, tg_y = [], []
    for name, runs in results.items():
        for r in runs:
            if r.get("n_prompt", 0) > 0:
                pp_x.append(r["n_prompt"])
                pp_y.append(r.get("avg_ts", 0))
            if r.get("n_gen", 0) > 0:
                tg_x.append(r["n_gen"])
                tg_y.append(r.get("avg_ts", 0))

    if pp_x:
        plt.figure()
        plt.plot(pp_x, pp_y, marker="o")
        plt.xlabel("Prompt tokens (-p)")
        plt.ylabel("tokens/s")
        plt.title("Prompt Performance")
        plt.savefig(out_dir / "prompt_perf.png")

    if tg_x:
        plt.figure()
        plt.plot(tg_x, tg_y, marker="o")
        plt.xlabel("Generated tokens (-n)")
        plt.ylabel("tokens/s")
        plt.title("Generation Performance")
        plt.savefig(out_dir / "gen_perf.png")


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="AMD LLM benchmark runner")
    parser.add_argument("config", help="JSON configuration file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("bench_runs") / timestamp
    out_dir.mkdir(parents=True)
    run_start = datetime.utcnow()

    # store system and version info
    exe_path = cfg.get("executable", "llama-bench")
    sys_info = gather_system_info(exe_path)
    (out_dir / "info.json").write_text(json.dumps(sys_info, indent=2))

    all_results: Dict[str, List[Dict]] = {}

    for test in cfg.get("tests", []):
        name = test.get("name", "run")
        args_list = test.get("args", [])
        exe = cfg.get("executable", "llama-bench")

        for p in cfg.get("p_values", []):
            cmd = [exe] + args_list + ["-p", str(p), "-n", "0", "-o", "jsonl"]
            run_name = f"{name}_p{p}"
            run_dir = out_dir / run_name
            run_dir.mkdir()
            results = run_single(cmd, run_dir)
            all_results.setdefault(name, []).extend(results)

        for n in cfg.get("n_values", []):
            cmd = [exe] + args_list + ["-n", str(n), "-p", "0", "-o", "jsonl"]
            run_name = f"{name}_n{n}"
            run_dir = out_dir / run_name
            run_dir.mkdir()
            results = run_single(cmd, run_dir)
            all_results.setdefault(name, []).extend(results)

    summary_file = out_dir / "results.json"
    summary_file.write_text(json.dumps(all_results, indent=2))

    generate_graph(all_results, out_dir)

    # simple markdown table
    md_lines = ["| Run | pp512 (t/s) | tg128 (t/s) | Max Mem (MiB) |", "| --- | --- | --- | --- |"]
    for name, runs in all_results.items():
        pp = next((r for r in runs if r.get("n_prompt") == 512), None)
        tg = next((r for r in runs if r.get("n_gen") == 128), None)
        pp_ts = f"{pp.get('avg_ts'):.2f} ± {pp.get('stddev_ts'):.2f}" if pp else "-"
        tg_ts = f"{tg.get('avg_ts'):.2f} ± {tg.get('stddev_ts'):.2f}" if tg else "-"
        md_lines.append(f"| {name} | {pp_ts} | {tg_ts} | - |")
    (out_dir / "summary.md").write_text("\n".join(md_lines))

    run_end = datetime.utcnow()
    run_info = {
        "start_time": run_start.isoformat() + "Z",
        "end_time": run_end.isoformat() + "Z",
        "duration_s": (run_end - run_start).total_seconds(),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))


def run_single(cmd: List[str], run_dir: Path) -> List[Dict]:
    """Run benchmark with watchers."""
    print(f"Running: {' '.join(cmd)}")
    start = datetime.utcnow()
    # watchers
    sys_mem = MetricWatcher("system_mem", "free --mebi | awk '/^Mem:/ {print $3}'")
    vram = MetricWatcher(
        "vram", "rocm-smi --showmeminfo vram --csv | awk -F, 'NR==2{print int($3/1048576)}'"
    )
    gtt = MetricWatcher(
        "gtt", "amdgpu_top -d | awk '/^[[:space:]]*GTT/{print int($4)}'"
    )
    sensors = SensorWatcher()

    for w in (sys_mem, vram, gtt, sensors):
        w.start()

    results = run_benchmark(cmd, run_dir)
    end = datetime.utcnow()

    for w in (sys_mem, vram, gtt, sensors):
        w.stop()
        w.join()

    metrics = {
        "system_mem_initial": sys_mem.initial,
        "system_mem_peak": sys_mem.max_value,
        "vram_initial": vram.initial,
        "vram_peak": vram.max_value,
        "gtt_initial": gtt.initial,
        "gtt_peak": gtt.max_value,
        "max_temp_c": sensors.max_temp,
        "max_power_w": sensors.max_power,
        "avg_power_w": sensors.avg_power(),
        "median_power_w": sensors.median_power(),
        "start_time": start.isoformat() + "Z",
        "end_time": end.isoformat() + "Z",
        "duration_s": (end - start).total_seconds(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # merge metrics into each result entry
    for r in results:
        r.update(metrics)

    return results


if __name__ == "__main__":
    main()
