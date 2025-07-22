#!/usr/bin/env python3

import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate

# MODELS to lookup/render
MODELS = {
    "dots.llm1.inst-UD-Q4_K_XL": {
        "name": "dots1 UD-Q4_K_XL",
        "arch": "dots1 MoE",
        "weights": 142,
        "active": 14,
        "pp": "vulkan_fa_b=256",
        "tg": "vulkan_fa_b=256",
        "line": {
            "color": (1.0, 0.0, 0.0, 0.9),
            "lw": 1,
            "ls": ":",
        },
    },
    "Hunyuan-A13B-Instruct-UD-Q6_K_XL": {
        "name": "Hunyuan-A13B UD-Q6_K_XL",
        "arch": "Hunyuan MoE",
        "weights": 80,
        "active": 13,
        "pp": "vulkan_fa",
        "tg": "vulkan_fa_b=256",
        "line": {
            "color": (0.7, 0.7, 0.7, 0.9),
            "lw": 1,
            "ls": ":",
        },
    },
    "Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL": {
        "name": "Llama 4 Scout UD-Q4_K_XL",
        "arch": "Llama 4 MoE",
        "weights": 109,
        "active": 17,
        "pp": "hip_hipblaslt",
        "tg": "vulkan_fa_b=256",
        "line": {
            "color": (0.0, 0.3, 0.9, 0.9),
            "lw": 1,
            "ls": ":",
        },
    },
    "Mistral-Small-3.1-24B-Instruct-2503-UD-Q4_K_XL": {
        "name": "Mistral Small 3.1 UD-Q4_K_XL",
        "arch": "Mistral 3",
        "weights": 24,
        "active": 24,
        "pp": "hip_hipblaslt",
        "tg": "vulkan_fa",
        "line": {
            "color": (0.9, 0.7, 0.0, 0.9),
            "lw": 1,
            "ls": "-",
        },
    },
    "Qwen3-30B-A3B-UD-Q4_K_XL": {
        "name": "Qwen 3 30B-A3B UD-Q4_K_XL",
        "arch": "Qwen 3 MoE",
        "weights": 30,
        "active": 3,
        "pp": "vulkan_fa",
        "tg": "vulkan_b=256",
        "line": {
            "color": (0.5, 0.5, 0.9, 0.9),
            "lw": 1,
            "ls": ":",
        },
    },
    "llama-2-7b.Q4_0": {
        "name": "Llama 2 7B Q4_0",
        "arch": "Llama 2",
        "weights": 7,
        "active": 7,
        "pp": "vulkan",
        "tg": "vulkan_fa",
        "line": {
            "color": (0.2, 0.5, 0.9, 0.9),
            "lw": 1,
            "ls": "-",
        },
    },
    "llama-2-7b.Q4_K_M": {
        "name": "Llama 2 7B Q4_K_M",
        "arch": "Llama 2",
        "weights": 7,
        "active": 7,
        "pp": "hip_hipblaslt",
        "tg": "vulkan_fa",
        "line": {
            "color": (0.1, 0.4, 1.0, 0.9),
            "lw": 1,
            "ls": "-",
        },
    },
    "shisa-v2-llama3.1-8b.i1-Q4_K_M": {
        "name": "Shisa V2 8B i1-Q4_K_M",
        "arch": "Llama 3",
        "weights": 8,
        "active": 8,
        "pp": "hip_hipblaslt",
        "tg": "vulkan_fa",
        "line": {
            "color": (0.8, 0.6, 0.8, 0.9),
            "lw": 1,
            "ls": "-",
        },
    },
    "shisa-v2-llama3.3-70b.i1-Q4_K_M": {
        "name": "Shisa V2 70B i1-Q4_K_M",
        "arch": "Llama 3",
        "weights": 70,
        "active": 70,
        "pp": "rocwmma",
        "tg": "vulkan_fa",
        "line": {
            "color": (0.7, 0.3, 0.7, 0.9),
            "lw": 1,
            "ls": "-",
        },
    },
}

def load_model_results(model_key):
    """Load results.jsonl for a given model."""
    model_dir = Path(model_key)
    results_file = model_dir / "results.jsonl"
    
    if not results_file.exists():
        print(f"Warning: No results file found for {model_key}")
        return None
    
    try:
        df = pd.read_json(results_file, orient='records', lines=True)
        return df
    except Exception as e:
        print(f"Error loading results for {model_key}: {e}")
        return None

def get_best_performance_for_backend(df, backend_config, mode):
    """Get performance for a specific backend configuration and mode."""
    if df is None or df.empty:
        return None, None
    
    # Filter for the specific backend configuration
    if backend_config == "vulkan_fa_b=256":
        mask = (df['build'].str.contains('vulkan', na=False)) & \
               (df['fa'] == '-fa 1') & \
               (df['b'] == '-b 256')
    elif backend_config == "vulkan_fa":
        mask = (df['build'].str.contains('vulkan', na=False)) & \
               (df['fa'] == '-fa 1') & \
               (df['b'] == '')
    elif backend_config == "vulkan_b=256":
        mask = (df['build'].str.contains('vulkan', na=False)) & \
               (df['fa'] == '') & \
               (df['b'] == '-b 256')
    elif backend_config == "vulkan":
        mask = (df['build'].str.contains('vulkan', na=False)) & \
               (df['fa'] == '') & \
               (df['b'] == '')
    elif backend_config == "hip_hipblaslt":
        mask = (df['build'].str.contains('hip', na=False)) & \
               (df['hipblaslt'] == '1')
    elif backend_config == "rocwmma":
        mask = (df['build'].str.contains('rocwmma', na=False))
    else:
        mask = df['build'].str.contains(backend_config, na=False)
    
    filtered_df = df[mask & (df['mode'] == mode)]
    
    if filtered_df.empty:
        return None, None
    
    # Get the performance values and memory info
    performance = filtered_df.groupby('value')['tokens_per_sec'].max()
    max_memory = filtered_df[['vram_peak_mib', 'gtt_peak_mib']].sum(axis=1).max()
    
    return performance, max_memory

def get_best_performance(df, model_info, mode):
    """Get the best performance for a model in the given mode."""
    if df is None or df.empty:
        return None, None
    
    # Get the best backend for this mode
    best_backend = model_info[mode]
    
    # Filter for the best backend configuration
    if best_backend == "vulkan_fa_b=256":
        mask = (df['build'].str.contains('vulkan', na=False)) & \
               (df['fa'] == '-fa 1') & \
               (df['b'] == '-b 256')
    elif best_backend == "vulkan_fa":
        mask = (df['build'].str.contains('vulkan', na=False)) & \
               (df['fa'] == '-fa 1') & \
               (df['b'] == '')
    elif best_backend == "vulkan_b=256":
        mask = (df['build'].str.contains('vulkan', na=False)) & \
               (df['fa'] == '') & \
               (df['b'] == '-b 256')
    elif best_backend == "vulkan":
        mask = (df['build'].str.contains('vulkan', na=False)) & \
               (df['fa'] == '') & \
               (df['b'] == '')
    elif best_backend == "hip_hipblaslt":
        mask = (df['build'].str.contains('hip', na=False)) & \
               (df['hipblaslt'] == '1')
    elif best_backend == "rocwmma":
        mask = (df['build'].str.contains('rocwmma', na=False))
    else:
        mask = df['build'].str.contains(best_backend, na=False)
    
    filtered_df = df[mask & (df['mode'] == mode)]
    
    if filtered_df.empty:
        return None, None
    
    # Get the performance values and memory info
    performance = filtered_df.groupby('value')['tokens_per_sec'].max()
    max_memory = filtered_df[['vram_peak_mib', 'gtt_peak_mib']].sum(axis=1).max()
    
    return performance, max_memory

def convert_backend_to_human_readable(backend_config):
    """Convert backend configuration to human readable format."""
    if backend_config.startswith("vulkan"):
        return "Vulkan"
    elif backend_config.startswith("hip"):
        return "HIP"
    elif backend_config.startswith("rocwmma"):
        return "HIP rocWMMA"
    else:
        return backend_config.upper()

def extract_flags(backend_config):
    """Extract flags from backend configuration."""
    flags = []
    if "fa" in backend_config:
        flags.append("fa=1")
    if "b=256" in backend_config:
        flags.append("b=256")
    if "hipblaslt" in backend_config:
        flags.append("hipBLASLt")
    return " ".join(flags) if flags else ""

def get_marker_for_backend(backend_config):
    """Get the appropriate marker for a backend configuration."""
    if backend_config.startswith("vulkan"):
        return "o"  # Circle for Vulkan
    elif backend_config.startswith("hip") or backend_config.startswith("rocwmma"):
        return "s"  # Square for HIP/rocWMMA
    else:
        return "o"  # Default to circle

def create_mode_table(all_results, mode, sort_by='weights', sort_descending=True):
    """Create a summary table for a specific mode (pp or tg) using optimal backend for that mode.
    
    Args:
        all_results: Dictionary of model results
        mode: 'pp' or 'tg' 
        sort_by: Column to sort by - 'weights', 'active', 'pp512', 'tg128', 'memory', or 'name'
        sort_descending: Whether to sort in descending order (True) or ascending (False)
    """
    rows = []
    
    # Initially sort by weights to maintain consistent ordering for processing
    sorted_models = sorted(MODELS.items(), key=lambda x: x[1]['weights'], reverse=True)
    
    for model_key, model_info in sorted_models:
        df = all_results.get(model_key)
        if df is None:
            continue
            
        # Get performance for the mode using its optimal backend
        mode_performance, mode_memory = get_best_performance(df, model_info, mode)
        
        if mode_performance is None:
            continue
        
        # For each table, we need to get both PP and TG performance from the SAME backend
        # The mode_performance only contains performance for the current mode
        # We need to get both PP and TG performance from the same backend configuration
        
        # Get the backend configuration for this mode
        backend_config = model_info[mode]
        
        # Get both PP and TG performance from the same backend
        pp_performance_same_backend, pp_memory_same_backend = get_best_performance_for_backend(df, backend_config, 'pp')
        tg_performance_same_backend, tg_memory_same_backend = get_best_performance_for_backend(df, backend_config, 'tg')
        
        # Use memory from whichever mode is available (they should be the same backend)
        backend_memory = pp_memory_same_backend if pp_memory_same_backend is not None else tg_memory_same_backend
        
        pp512_val = pp_performance_same_backend.get(512, None) if pp_performance_same_backend is not None else None
        tg128_val = tg_performance_same_backend.get(128, None) if tg_performance_same_backend is not None else None
        
        pp512 = f"{round(float(pp512_val), 1):.1f}" if pp512_val is not None else "-"
        tg128 = f"{round(float(tg128_val), 1):.1f}" if tg128_val is not None else "-"
        
        # Use the backend configuration for this mode
        backend_config = model_info[mode]
        
        row = {
            'Model Name': model_info['name'],
            'Architecture': model_info['arch'],
            'Weights (B)': model_info['weights'],
            'Active (B)': model_info['active'],
            'Backend': convert_backend_to_human_readable(backend_config),
            'Flags': extract_flags(backend_config),
            'pp512': pp512,
            'tg128': tg128,
            'Memory (Max MiB)': f"{backend_memory:.0f}" if backend_memory and backend_memory > 0 else "-",
        }
        
        rows.append(row)
    
    # Sort the final table by the specified column
    if sort_by == 'weights':
        rows.sort(key=lambda x: x['Weights (B)'], reverse=sort_descending)
    elif sort_by == 'active':
        rows.sort(key=lambda x: x['Active (B)'], reverse=sort_descending)
    elif sort_by == 'pp512':
        rows.sort(key=lambda x: float(x['pp512']) if x['pp512'] != '-' else 0, reverse=sort_descending)
    elif sort_by == 'tg128':
        rows.sort(key=lambda x: float(x['tg128']) if x['tg128'] != '-' else 0, reverse=sort_descending)
    elif sort_by == 'memory':
        rows.sort(key=lambda x: float(x['Memory (Max MiB)']) if x['Memory (Max MiB)'] != '-' else 0, reverse=sort_descending)
    elif sort_by == 'name':
        rows.sort(key=lambda x: x['Model Name'], reverse=sort_descending)
    
    return rows

def create_summary_plot(all_results, mode, output_path):
    """Create a summary plot for all models."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort models by weights (descending) for consistent ordering
    sorted_models = sorted(MODELS.items(), key=lambda x: x[1]['weights'], reverse=True)
    
    for model_key, model_info in sorted_models:
        df = all_results.get(model_key)
        if df is None:
            continue
            
        performance, _ = get_best_performance(df, model_info, mode)
        if performance is None or performance.empty:
            continue
        
        style = model_info['line'].copy()
        # Override marker based on the backend used for this mode
        backend_config = model_info[mode]
        style['marker'] = get_marker_for_backend(backend_config)
        
        ax.plot(
            performance.index,
            performance.values,
            label=model_info['name'],
            **style
        )
    
    if ax.lines:
        ax.set_title(f"Model Performance Comparison - {mode.upper()}")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Tokens per Second")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(True, alpha=0.3, linestyle=":")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    plt.close(fig)

def generate_readme():
    """Generate the README.md file with results."""
    
    # SORTING CONFIGURATION - Modify these to change table sorting
    # Available options for SORT_BY:
    #   'weights' - Sort by total model weights (good for showing model size progression)
    #   'active'  - Sort by active parameters (good for MoE model comparison)  
    #   'pp512'   - Sort by prompt processing performance (shows fastest PP models first)
    #   'tg128'   - Sort by text generation performance (shows fastest TG models first)
    #   'memory'  - Sort by memory usage (shows most/least memory intensive)
    #   'name'    - Sort alphabetically by model name
    
    PP_SORT_BY = 'pp512'        # Options: 'weights', 'active', 'pp512', 'tg128', 'memory', 'name'
    PP_SORT_DESCENDING = True     # True for descending, False for ascending
    
    TG_SORT_BY = 'tg128'        # Options: 'weights', 'active', 'pp512', 'tg128', 'memory', 'name'  
    TG_SORT_DESCENDING = True     # True for descending, False for ascending
    
    # Load all model results
    all_results = {}
    for model_key in MODELS.keys():
        all_results[model_key] = load_model_results(model_key)
    
    # Create separate tables for PP and TG modes with configurable sorting
    pp_rows = create_mode_table(all_results, 'pp', PP_SORT_BY, PP_SORT_DESCENDING)
    tg_rows = create_mode_table(all_results, 'tg', TG_SORT_BY, TG_SORT_DESCENDING)
    
    # Create plots
    create_summary_plot(all_results, 'pp', 'summary-results-pp.png')
    create_summary_plot(all_results, 'tg', 'summary-results-tg.png')
    
    # Read existing README.md if it exists
    readme_path = Path('README.md')
    if readme_path.exists():
        existing_content = readme_path.read_text()
        
        # Find the ## Results section and everything before it
        results_start = existing_content.find('## Results')
        if results_start != -1:
            # Find the next ## section after Results
            next_section_start = existing_content.find('\n## ', results_start + 1)
            if next_section_start != -1:
                # Keep content before Results and after the next section
                before_results = existing_content[:results_start]
                after_results = existing_content[next_section_start:]
            else:
                # No section after Results, keep only content before
                before_results = existing_content[:results_start]
                after_results = ""
        else:
            # No Results section exists, append to end
            before_results = existing_content.rstrip() + "\n\n"
            after_results = ""
    else:
        # No README exists, create header
        before_results = """# LLM Benchmark Results

This directory contains benchmark results for various language models tested on AMD hardware.

"""
        after_results = ""
    
    # Generate new Results section
    results_content = """## Results

### Prompt Processing (pp) Performance
![PP Performance](summary-results-pp.png)

"""
    
    if pp_rows:
        pp_table = tabulate(pp_rows, headers='keys', tablefmt='github', floatfmt=(".0f", ".0f", ".0f", ".0f", "", "", ".1f", ".1f", ".0f"))
        results_content += pp_table + "\n\n"
    
    results_content += """### Text Generation (tg) Performance
![TG Performance](summary-results-tg.png)

"""
    
    if tg_rows:
        tg_table = tabulate(tg_rows, headers='keys', tablefmt='github', floatfmt=(".0f", ".0f", ".0f", ".0f", "", "", ".1f", ".1f", ".0f"))
        results_content += tg_table + "\n\n"
    
    '''
    results_content += """### Performance Notes

- **pp512**: Prompt processing performance at 512 tokens
- **tg128**: Text generation performance at 128 tokens  
- **Memory**: Peak GPU memory usage (VRAM + GTT)
- Models are ordered by parameter count (descending)
- Backend configurations are optimized per model and mode
- Each table shows performance using the optimal backend for that mode
- Benchmarks run on AMD hardware with various backend configurations

"""
    '''
    
    # Combine all content
    full_content = before_results + results_content + after_results
    
    # Write README.md
    readme_path.write_text(full_content)
    
    print("README.md updated successfully!")
    print("Generated summary-results-pp.png and summary-results-tg.png")

if __name__ == "__main__":
    generate_readme()
