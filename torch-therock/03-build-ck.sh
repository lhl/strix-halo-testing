#!/usr/bin/env bash
set -euo pipefail

# Incremental by default; use --clean to force a full rebuild.
# Allows overriding install prefix and GPU targets:
#   ./03-build-ck.sh --prefix="$ROCM_PATH" --targets=gfx1151

CLEAN=0
PREFIX="${CK_INSTALL_PREFIX:-${ROCM_PATH:-${CONDA_PREFIX:-}}}"
TARGETS="${CK_GPU_TARGETS:-gfx1151}"

for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --prefix=*) PREFIX="${arg#*=}" ;;
    --targets=*) TARGETS="${arg#*=}" ;;
    *) echo "[ck] Warning: unknown arg '$arg' (ignored)" ;;
  esac
done

if [[ -z "${PREFIX}" ]]; then
  PREFIX="/opt/rocm"
fi

echo "[ck] Install prefix: ${PREFIX}"
echo "[ck] GPU targets:    ${TARGETS}"
if [[ ${CLEAN} -eq 1 ]]; then
  echo "[ck] Clean build requested via --clean"
else
  echo "[ck] Resuming existing checkout/build if present (use --clean to rebuild)"
fi

if [[ ! -d composable_kernel/.git ]]; then
  echo "[ck] Cloning ROCm/composable_kernel..."
  git clone https://github.com/ROCm/composable_kernel
else
  echo "[ck] Using existing composable_kernel checkout"
fi
cd composable_kernel

if [[ ${CLEAN} -eq 1 && -d build ]]; then
  echo "[ck] Removing existing build directory"
  rm -rf build
fi
mkdir -p build
cd build

# Prefer /opt/rocm toolchain, but allow override via env if needed.
CLANG_BIN="/opt/rocm/llvm/bin"
CC_BIN="${CC:-${CLANG_BIN}/amdclang}"
CXX_BIN="${CXX:-${CLANG_BIN}/amdclang++}"
HIP_CXX_BIN="${HIP_CLANG_PATH:-${CLANG_BIN}}/clang++"

echo "[ck] Configuring CMake..."
CC="${CC_BIN}" CXX="${CXX_BIN}" cmake \
  -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="${PREFIX}" \
  -D CMAKE_CXX_COMPILER="${CXX_BIN}" \
  -D CMAKE_HIP_COMPILER="${HIP_CXX_BIN}" \
  -D GPU_TARGETS="${TARGETS}" \
  -D BUILD_DEV=OFF \
  -D CMAKE_CXX_FLAGS="-U_GLIBCXX_ASSERTIONS -Wno-error=old-style-cast" \
  ..

echo "[ck] Building..."
ninja -j"${NPROC:-$(nproc)}"

echo "[ck] Installing to ${PREFIX}..."
if ninja install; then
  echo "[ck] Install complete"
else
  echo "[ck] Regular install failed, retrying with sudo..."
  sudo ninja install
fi
