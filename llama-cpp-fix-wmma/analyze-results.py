#!/usr/bin/env python3
"""Parse llama-bench JSONL outputs and render summary tables."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - tabulate may not be installed
    tabulate = None  # type: ignore[assignment]

try:
    from rich.console import Console
    console = Console()
except ImportError:  # pragma: no cover - rich may not be installed
    console = None  # type: ignore[assignment]


class BenchResult:
    def __init__(self, payload: dict, source: Path) -> None:
        self.payload = payload
        self.source = source

    @property
    def model(self) -> str:
        return self.payload.get("model_type", "unknown")

    @property
    def size_bytes(self) -> int:
        return int(self.payload.get("model_size", 0))

    @property
    def params(self) -> int:
        return int(self.payload.get("model_n_params", 0))

    @property
    def avg_ts(self) -> float:
        return float(self.payload.get("avg_ts", float("nan")))

    @property
    def std_ts(self) -> float:
        return float(self.payload.get("stddev_ts", float("nan")))

    @property
    def n_prompt(self) -> int:
        return int(self.payload.get("n_prompt", 0))

    @property
    def n_gen(self) -> int:
        return int(self.payload.get("n_gen", 0))

    @property
    def depth(self) -> int:
        return int(self.payload.get("n_depth", 0))

    def test_label(self) -> str:
        parts: List[str] = []
        if self.n_prompt > 0:
            parts.append(f"pp{self.n_prompt}")
        elif self.n_gen > 0:
            parts.append(f"tg{self.n_gen}")
        else:
            parts.append("unknown")

        if self.depth > 0:
            parts.append(f"@ d{self.depth}")
        return " ".join(parts)

    def size_display(self) -> str:
        mib = self.size_bytes / (1024 ** 2)
        return f"{mib:7.2f} MiB"

    def params_display(self) -> str:
        return human_params(self.params)

    def ts_display(self) -> str:
        if math.isnan(self.avg_ts):
            return "n/a"
        if math.isnan(self.std_ts) or self.std_ts < 1e-6:
            return f"{self.avg_ts:7.2f}"
        return f"{self.avg_ts:7.2f} ± {self.std_ts:.2f}"

    def key(self) -> Tuple[str, int, int, int]:
        return (self.model, self.n_prompt, self.n_gen, self.depth)


def human_params(count: int) -> str:
    units: Tuple[Tuple[int, str], ...] = (
        (10**12, "T"),
        (10**9, "B"),
        (10**6, "M"),
    )
    for threshold, suffix in units:
        if count >= threshold:
            value = count / threshold
            return f"{value:6.2f} {suffix}"
    return f"{count}"


def read_results(path: Path) -> List[BenchResult]:
    results: List[BenchResult] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            results.append(BenchResult(payload, path))
    return results


def sort_results(results: Iterable[BenchResult]) -> List[BenchResult]:
    return sorted(
        results,
        key=lambda r: (r.depth, 0 if r.n_prompt > 0 else 1, -r.n_prompt, -r.n_gen),
    )


def pair_results(
    hip_results: Iterable[BenchResult], roc_results: Iterable[BenchResult]
) -> List[Tuple[BenchResult, BenchResult]]:
    hip_map: Dict[Tuple[str, int, int, int], BenchResult] = {
        res.key(): res for res in hip_results
    }
    pairs: List[Tuple[BenchResult, BenchResult]] = []
    missing_baseline: List[BenchResult] = []
    matched_keys: set[Tuple[str, int, int, int]] = set()

    for roc in roc_results:
        key = roc.key()
        hip = hip_map.get(key)
        if hip is None:
            missing_baseline.append(roc)
            continue
        pairs.append((hip, roc))
        matched_keys.add(key)

    if missing_baseline:
        unique_keys = {item.key(): item for item in missing_baseline}
        if console:
            console.print("Warning: no baseline entry for the following rocWMMA tests:")
            for item in unique_keys.values():
                console.print(f"  {item.model} - {item.test_label()} (depth {item.depth})")
        else:
            print("Warning: no baseline entry for the following rocWMMA tests:")
            for item in unique_keys.values():
                print(f"  {item.model} - {item.test_label()} (depth {item.depth})")

    missing_roc = [
        hip for key, hip in hip_map.items() if key not in matched_keys
    ]
    if missing_roc:
        unique_keys = {item.key(): item for item in missing_roc}
        if console:
            console.print("Warning: no rocWMMA entry for the following baseline tests:")
            for item in unique_keys.values():
                console.print(f"  {item.model} - {item.test_label()} (depth {item.depth})")
        else:
            print("Warning: no rocWMMA entry for the following baseline tests:")
            for item in unique_keys.values():
                print(f"  {item.model} - {item.test_label()} (depth {item.depth})")

    return sorted(
        pairs,
        key=lambda t: (
            t[0].depth,
            0 if t[0].n_prompt > 0 else 1,
            -t[0].n_prompt,
            -t[0].n_gen,
        ),
    )


def percent_delta(base: float, variant: float, colorize: bool = False) -> str:
    if math.isnan(base) or base == 0 or math.isnan(variant):
        return "n/a"
    delta = (variant / base) - 1.0
    percent_str = f"{delta * 100:6.2f}%"

    if not colorize or not console:
        return percent_str

    # Add pastel colors: green for positive, red for negative
    if delta > 0:
        # Pastel green
        return f"[pale_green3]{percent_str}[/pale_green3]"
    elif delta < 0:
        # Pastel red/pink
        return f"[light_pink3]{percent_str}[/light_pink3]"
    else:
        return percent_str


def build_rows(pairs: Iterable[Tuple[BenchResult, BenchResult]]) -> List[List[str]]:
    rows = []
    for base_res, variant_res in pairs:
        rows.append(
            [
                base_res.model,
                base_res.size_display(),
                base_res.params_display(),
                base_res.test_label(),
                base_res.ts_display(),
                variant_res.ts_display(),
                percent_delta(base_res.avg_ts, variant_res.avg_ts),
            ]
        )
    return rows


def colorize_line(line: str) -> str:
    """Add colors to delta percentages in a table line."""
    if not console:
        return line

    # Skip header and separator lines
    if not '|' in line or line.strip().startswith('|---') or 'Δ%' in line:
        return line

    # Match percentage values in the last column
    # Split by last pipe to get the delta column (strip trailing pipe first)
    parts = line.rstrip('|').rsplit('|', 1)
    if len(parts) != 2:
        return line

    left, right = parts

    # Look for percentage pattern: optional spaces, optional minus, digits, dot, digits, %
    match = re.search(r'(-?\d+\.\d+)%', right)
    if match:
        value_str = match.group(1)
        try:
            value = float(value_str)
            # Neutral within margin of error (<0.5%)
            if abs(value) < 0.5:
                colored = right.replace(f'{value_str}%', f'[sky_blue1]{value_str}%[/sky_blue1]')
            elif value < 0:
                # Negative = regression = pink/red
                colored = right.replace(f'{value_str}%', f'[light_pink3]{value_str}%[/light_pink3]')
            else:
                # Positive = improvement = green
                colored = right.replace(f'{value_str}%', f'[pale_green3]{value_str}%[/pale_green3]')
            return left + '|' + colored + '|'
        except ValueError:
            pass

    return line


def print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    if not rows:
        return

    if console:
        console.print(title)
        console.print("-" * len(title))
    else:
        print(title)
        print("-" * len(title))

    if tabulate:
        table_output = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            colalign=("left", "right", "right", "left", "right", "right", "right"),
        )

        # Colorize each line
        if console:
            for line in table_output.split('\n'):
                colored_line = colorize_line(line)
                console.print(colored_line, markup=True, highlight=False)
        else:
            print(table_output)
    else:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for idx, value in enumerate(row):
                col_widths[idx] = max(col_widths[idx], len(value))

        def fmt_row(values: List[str]) -> str:
            parts = []
            for idx, value in enumerate(values):
                width = col_widths[idx]
                if idx in (1, 2, 4, 5, 6):
                    parts.append(value.rjust(width))
                else:
                    parts.append(value.ljust(width))
            return " | ".join(parts)

        if console:
            console.print(fmt_row(headers))
            console.print(fmt_row(["-" * w for w in col_widths]))
            for row in rows:
                console.print(fmt_row(row))
        else:
            print(fmt_row(headers))
            print(fmt_row(["-" * w for w in col_widths]))
            for row in rows:
                print(fmt_row(row))

    if console:
        console.print()
    else:
        print()


def render_comparison(
    base: List[BenchResult],
    variant: List[BenchResult],
    base_label: str,
    variant_label: str,
    model_tag: Optional[str] = None,
) -> None:
    pairs = pair_results(base, variant)
    if not pairs:
        if console:
            console.print("No comparable entries found.")
        else:
            print("No comparable entries found.")
        return

    # Shorten labels for table headers
    base_short = "HIP" if "hip" in base_label.lower() else base_label
    variant_short = "WMMA" if "wmma" in variant_label.lower() or "roc" in variant_label.lower() else variant_label

    headers = [
        "model",
        "size",
        "params",
        "test",
        base_short,
        variant_short,
        "Δ%",
    ]

    title = f"{base_label} vs {variant_label}"
    if model_tag:
        if console:
            console.print(f"Model: {model_tag}")
        else:
            print(f"Model: {model_tag}")
    if console:
        console.print(title)
        console.print("=" * len(title))
        console.print()
    else:
        print(title)
        print("=" * len(title))
        print()

    groups = [
        (
            "Prefill (pp)",
            [pair for pair in pairs if pair[0].n_prompt > 0 or pair[1].n_prompt > 0],
        ),
        (
            "Decode (tg)",
            [pair for pair in pairs if pair[0].n_gen > 0 or pair[1].n_gen > 0],
        ),
        (
            "Other",
            [
                pair
                for pair in pairs
                if (pair[0].n_prompt == 0 and pair[1].n_prompt == 0)
                and (pair[0].n_gen == 0 and pair[1].n_gen == 0)
            ],
        ),
    ]

    for group_title, group_pairs in groups:
        rows = build_rows(group_pairs)
        if not rows:
            continue
        print_table(group_title, headers, rows)


def split_tag_from_stem(stem: str) -> str:
    """Return leading tag before first '.' in a filename stem."""
    return stem.split(".", 1)[0]


def strip_tag_from_label(label: str, tag: Optional[str]) -> str:
    if tag and label.startswith(tag + "."):
        return label[len(tag) + 1 :]
    return label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare llama-bench JSONL results.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help=(
            "Model tag prefix for default files, e.g. 'llama32-1b' to use "
            "'llama32-1b.default-hip.jsonl' and 'llama32-1b.default-rocwmma.jsonl'."
        ),
    )
    parser.add_argument(
        "--hip",
        type=Path,
        default=None,
        help="HIP (baseline) JSONL file (overrides --tag default)",
    )
    parser.add_argument(
        "--rocwmma",
        type=Path,
        default=None,
        help="rocWMMA JSONL file to compare (overrides --tag default)",
    )
    parser.add_argument(
        "variants",
        nargs="*",
        type=Path,
        help="Additional JSONL files to compare against the baselines",
    )
    args = parser.parse_args()

    # Resolve defaults based on tag if explicit files not provided.
    hip_path: Optional[Path] = args.hip
    roc_path: Optional[Path] = args.rocwmma

    model_tag: Optional[str] = args.tag

    # If tag not provided, try to infer from provided file names (stem before first '.')
    if model_tag is None:
        for p in (hip_path, roc_path) + tuple(args.variants):
            if p is not None:
                try:
                    model_tag = split_tag_from_stem(p.stem)
                    break
                except Exception:
                    continue

    # If a tag is available, use it to fill missing defaults
    if hip_path is None and model_tag is not None:
        hip_path = Path(f"{model_tag}.default-hip.jsonl")
    if roc_path is None and model_tag is not None:
        roc_path = Path(f"{model_tag}.default-rocwmma.jsonl")

    if hip_path is None or roc_path is None:
        parser.error("either provide --tag, or specify both --hip and --rocwmma files")

    if not hip_path.exists():
        parser.error(f"baseline file not found: {hip_path}")
    if not roc_path.exists():
        parser.error(f"rocWMMA file not found: {roc_path}")
    for variant_path in args.variants:
        if not variant_path.exists():
            parser.error(f"variant file not found: {variant_path}")

    hip_results = sort_results(read_results(hip_path))
    roc_results = sort_results(read_results(roc_path))
    hip_label_full = hip_path.stem
    roc_label_full = roc_path.stem
    # For titles, strip the tag to avoid verbosity in column headers/titles.
    hip_label = strip_tag_from_label(hip_label_full, model_tag)
    roc_label = strip_tag_from_label(roc_label_full, model_tag)

    render_comparison(hip_results, roc_results, hip_label, roc_label, model_tag=model_tag)

    for variant_path in args.variants:
        variant_results = sort_results(read_results(variant_path))
        variant_label = strip_tag_from_label(variant_path.stem, model_tag)
        if console:
            console.print()
        else:
            print()
        render_comparison(hip_results, variant_results, hip_label, variant_label, model_tag=model_tag)
        if console:
            console.print()
        else:
            print()
        render_comparison(roc_results, variant_results, roc_label, variant_label, model_tag=model_tag)


if __name__ == "__main__":
    main()
