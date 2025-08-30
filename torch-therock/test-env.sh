#!/usr/bin/env bash


# Step 2: Initialize and activate environment
echo "Activating therock environment..."

# Initialize conda/mamba in this shell session
if [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    # Try to find conda installation
    CONDA_BASE=$(find /opt /usr/local $HOME -name "etc/profile.d/conda.sh" 2>/dev/null | head -1 | xargs dirname | xargs dirname | xargs dirname 2>/dev/null)
    if [ -n "$CONDA_BASE" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "✗ Error: Could not find conda installation!"
        echo "Please ensure conda/mamba is installed and accessible."
        exit 1
    fi
fi

# Initialize mamba if available
if [ -f "$HOME/mambaforge/etc/profile.d/mamba.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/mamba.sh"
fi

conda activate therock


# PATH
PATH="$CONDA_PREFIX/bin:$PATH"
ROCM_BIN_PATH=$(rocm-sdk path --bin 2>/dev/null)

conda env config vars unset PATH
echo "Original Path: $PATH"
echo "New Path: $CONDA_PREFIX/bin:$ROCM_BIN_PATH:\$PATH"
conda env config vars set PATH="$CONDA_PREFIX/bin:$ROCM_BIN_PATH:\$PATH"
