mamba create -n torch python=3.12
# sudo dnf install xz-devel
mamba install liblzma-devel
pip install ninja

git clone https://github.com/ROCm/aotriton
cd aotriton
git submodule sync && git submodule update --init --recursive --force
mkdir build && cd build

cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_GPU_BUILD_TIMEOUT=0 -DAOTRITON_TARGET_ARCH="gfx1100;gfx1151" -G Ninja

ninja install
