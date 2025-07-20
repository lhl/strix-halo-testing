#!/bin/bash 

# if ! mamba env list | grep -q "^torch "; then
#     mamba create -n torch python=3.12
# fi
# mamba activate torch
mamba install xz -y

pip install uv
uv pip install ninja

# proper version installed by aotriton 
# uv pip install triton
