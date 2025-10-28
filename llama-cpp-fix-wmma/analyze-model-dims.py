#!/usr/bin/env python3
"""
Analyze head dimensions from popular model architectures to optimize TILE kernel pruning.
"""

import json
from pathlib import Path
from collections import defaultdict

# Known model configs from HuggingFace and llama.cpp documentation
POPULAR_MODELS = {
    # LLaMA family
    "llama-3.2-1B": {"hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 8},
    "llama-3.2-3B": {"hidden_size": 3072, "num_attention_heads": 24, "num_key_value_heads": 8},
    "llama-3.1-8B": {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8},
    "llama-3.1-70B": {"hidden_size": 8192, "num_attention_heads": 64, "num_key_value_heads": 8},
    "llama-3.1-405B": {"hidden_size": 16384, "num_attention_heads": 128, "num_key_value_heads": 8},
    "llama-2-7B": {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 32},
    "llama-2-13B": {"hidden_size": 5120, "num_attention_heads": 40, "num_key_value_heads": 40},
    "llama-2-70B": {"hidden_size": 8192, "num_attention_heads": 64, "num_key_value_heads": 8},

    # Mistral family
    "mistral-7B-v0.3": {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8},
    "mixtral-8x7B": {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8},
    "mixtral-8x22B": {"hidden_size": 6144, "num_attention_heads": 48, "num_key_value_heads": 8},

    # Qwen family
    "qwen2-0.5B": {"hidden_size": 896, "num_attention_heads": 14, "num_key_value_heads": 2},
    "qwen2-1.5B": {"hidden_size": 1536, "num_attention_heads": 12, "num_key_value_heads": 2},
    "qwen2-7B": {"hidden_size": 3584, "num_attention_heads": 28, "num_key_value_heads": 4},
    "qwen2-72B": {"hidden_size": 8192, "num_attention_heads": 64, "num_key_value_heads": 8},
    "qwen2.5-coder-32B": {"hidden_size": 5120, "num_attention_heads": 40, "num_key_value_heads": 8},

    # Phi family
    "phi-3-mini-4k": {"hidden_size": 3072, "num_attention_heads": 32, "num_key_value_heads": 32},
    "phi-3-medium-4k": {"hidden_size": 5120, "num_attention_heads": 40, "num_key_value_heads": 40},
    "phi-3.5-mini": {"hidden_size": 3072, "num_attention_heads": 32, "num_key_value_heads": 32},

    # Gemma family
    "gemma-2-2B": {"hidden_size": 2304, "num_attention_heads": 8, "num_key_value_heads": 4},
    "gemma-2-9B": {"hidden_size": 3584, "num_attention_heads": 16, "num_key_value_heads": 8},
    "gemma-2-27B": {"hidden_size": 4608, "num_attention_heads": 32, "num_key_value_heads": 16},

    # DeepSeek family
    "deepseek-v2-lite": {"hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 2},
    "deepseek-coder-v2": {"hidden_size": 5120, "num_attention_heads": 128, "num_key_value_heads": 128},

    # Other notable models
    "command-r-35B": {"hidden_size": 8192, "num_attention_heads": 64, "num_key_value_heads": 8},
    "Yi-1.5-9B": {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 4},
}

def compute_head_dims(config):
    """Compute head dimensions from model config."""
    hidden = config["hidden_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]

    # Standard transformer: head_dim = hidden / n_heads
    head_dim_q = hidden // n_heads
    head_dim_kv = hidden // n_heads  # Usually same, but some models differ

    return {
        "head_dim_q": head_dim_q,
        "head_dim_kv": head_dim_kv,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "gqa_ratio": n_heads // n_kv_heads if n_kv_heads > 0 else 1,
    }

def main():
    print("Popular Model Head Dimensions (for TILE kernel pruning)")
    print("=" * 80)

    dim_counts = defaultdict(int)
    gqa_ratios = set()

    for model_name, config in sorted(POPULAR_MODELS.items()):
        dims = compute_head_dims(config)
        head_dim = dims["head_dim_q"]
        dim_counts[head_dim] += 1
        gqa_ratios.add(dims["gqa_ratio"])

        print(f"{model_name:30s}  head_dim={head_dim:3d}  "
              f"n_heads={dims['n_heads']:3d}  n_kv_heads={dims['n_kv_heads']:3d}  "
              f"GQA_ratio={dims['gqa_ratio']:2d}")

    print("\n" + "=" * 80)
    print("Head Dimension Statistics:")
    print("-" * 80)
    for dim in sorted(dim_counts.keys()):
        count = dim_counts[dim]
        bar = "█" * (count * 2)
        print(f"  {dim:3d}: {count:2d} models {bar}")

    print("\n" + "=" * 80)
    print("GQA Ratios (affects ncols2 in TILE kernels):")
    print("-" * 80)
    for ratio in sorted(gqa_ratios):
        print(f"  {ratio:2d}x")

    print("\n" + "=" * 80)
    print("Recommended TILE kernel DV values to keep (sorted by frequency):")
    print("-" * 80)
    top_dims = sorted(dim_counts.items(), key=lambda x: (-x[1], x[0]))
    cumulative = 0
    total = len(POPULAR_MODELS)
    for dim, count in top_dims[:10]:
        cumulative += count
        coverage = (cumulative / total) * 100
        print(f"  {dim:3d} ({count:2d} models, {coverage:5.1f}% cumulative coverage)")

    print("\n" + "=" * 80)
    print("Current RDNA config table coverage:")
    print("-" * 80)
    rdna_configs = [40, 64, 80, 96, 112, 128, 256, 576]
    print(f"  Configured: {rdna_configs}")

    unique_dims = sorted(set(dim_counts.keys()))
    missing = [d for d in unique_dims if d not in rdna_configs]
    if missing:
        print(f"  Missing:    {missing}")
        for d in missing:
            print(f"    -> {d:3d} used by {dim_counts[d]} model(s)")
    else:
        print("  ✓ All popular model dimensions are covered!")

if __name__ == "__main__":
    main()
