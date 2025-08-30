#!/usr/bin/env bash

# pyaotriton
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp aotriton/build/install_dir/lib/pyaotriton.cpython-312-x86_64-linux-gnu.so $SITE_PACKAGES/


# PyTorch?
cp aotriton/build/install_dir/lib/pyaotriton.cpython-312-x86_64-linux-gnu.so $SITE_PACKAGES/torch/lib/
cp aotriton/build/install_dir/lib/libaotriton_v2.so.0.11.0 $SITE_PACKAGES/torch/lib/
mkdir -p $SITE_PACKAGES/torch/lib/aotriton.images
cp -r aotriton/build/install_dir/lib/aotriton.images/* $SITE_PACKAGES/torch/lib/aotriton.images/

cd $SITE_PACKAGES/torch/lib && ln -sf libaotriton_v2.so.0.11.0 libaotriton_v2.so
