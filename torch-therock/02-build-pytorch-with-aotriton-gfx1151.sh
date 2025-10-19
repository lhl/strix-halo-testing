#!/bin/bash

# Build PyTorch with AOTriton support for gfx1151
# This script uses your prebuilt aotriton installation
#
# Usage:
#   ./02-build-pytorch-with-aotriton-gfx1151.sh [--continue STAGE] [--help]
#
# Stages:
#   checkout   - Skip to repository checkout (after environment check)
#   rocm       - Skip to ROCm installation 
#   triton     - Skip to triton build
#   pytorch    - Skip to PyTorch build
#   audio      - Skip to torchaudio build
#   vision     - Skip to torchvision build

set -euo pipefail

SCRIPT_DIR="$(cd $(dirname $0) && pwd)"
THEROCK_DIR="$SCRIPT_DIR/TheRock"
PYTORCH_BUILD_DIR="$THEROCK_DIR/external-builds/pytorch"
CONTINUE_FROM=""

# Parse command line arguments
# Supported: --continue <stage> | --skip-audio | --skip-vision | --python-exe <path> | --help
SKIP_AUDIO=0
SKIP_VISION=0
PYTHON_EXE=${PYTHON_EXE:-python}
while [[ $# -gt 0 ]]; do
    case $1 in
        --continue)
            CONTINUE_FROM="$2"
            shift 2
            ;;
        --python-exe)
            PYTHON_EXE="$2"
            shift 2
            ;;
        --skip-audio)
            SKIP_AUDIO=1
            shift 1
            ;;
        --skip-vision)
            SKIP_VISION=1
            shift 1
            ;;
        --help|-h)
            echo "Build PyTorch with AOTriton support for gfx1151"
            echo ""
            echo "Usage: $0 [--continue STAGE] [--skip-audio] [--skip-vision] [--python-exe PATH]"
            echo ""
            echo "Available stages to continue from:"
            echo "  checkout   - Skip to repository checkout (after environment check)"
            echo "  rocm       - Skip to ROCm installation"
            echo "  triton     - Skip to triton build"
            echo "  pytorch    - Skip to PyTorch build"  
            echo "  audio      - Skip to torchaudio build"
            echo "  vision     - Skip to torchvision build"
            echo ""
            echo "Optional flags:"
            echo "  --skip-audio    Skip building torchaudio"
            echo "  --skip-vision   Skip building torchvision"
            echo "  --python-exe    Python interpreter to run build steps (default: python)"
            echo ""
            echo "Examples:"
            echo "  $0                             # Full build from start"
            echo "  $0 --continue pytorch         # Skip to PyTorch build"
            echo "  $0 --skip-audio --skip-vision # Build torch only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
    echo "Error: Python executable '$PYTHON_EXE' not found in PATH."
    echo "Set PYTHON_EXE to the interpreter from your prepared environment."
    exit 1
fi

# Function to check if we should skip to a specific stage
should_skip_to() {
    local stage="$1"
    if [ -n "$CONTINUE_FROM" ]; then
        case "$CONTINUE_FROM" in
            checkout) [[ "$stage" =~ ^(venv)$ ]] ;;
            rocm) [[ "$stage" =~ ^(venv|checkout)$ ]] ;;
            triton) [[ "$stage" =~ ^(venv|checkout|rocm)$ ]] ;;
            pytorch) [[ "$stage" =~ ^(venv|checkout|rocm|triton)$ ]] ;;
            audio) [[ "$stage" =~ ^(venv|checkout|rocm|triton|pytorch)$ ]] ;;
            vision) [[ "$stage" =~ ^(venv|checkout|rocm|triton|pytorch|audio)$ ]] ;;
            *) echo "Unknown continue stage: $CONTINUE_FROM"; exit 1 ;;
        esac
    else
        false
    fi
}

# Function to display current stage
show_stage() {
    echo ""
    echo "======================================="
    echo "STAGE: $1"
    echo "======================================="
}

# Check current state and suggest continue points
check_build_state() {
    echo "=== Checking current build state ==="
    local suggestions=()
    
    if [ -d "$PYTORCH_BUILD_DIR/pytorch" ] && [ -d "$PYTORCH_BUILD_DIR/triton" ]; then
        suggestions+=("checkout (repos already checked out)")
    fi
    
    # Check if ROCm is installed by looking for recent pip installs
    if "$PYTHON_EXE" -m pip show rocm-sdk >/dev/null 2>&1; then
        suggestions+=("rocm (ROCm SDK appears to be installed)")
    fi
    
    # Check for built wheels
    if ls "$HOME/tmp/pyout/"pytorch-triton-rocm-*.whl >/dev/null 2>&1; then
        suggestions+=("pytorch (triton wheel exists)")
    fi
    if ls "$HOME/tmp/pyout/"torch-*.whl >/dev/null 2>&1; then
        suggestions+=("audio (torch wheel exists)")
    fi
    if ls "$HOME/tmp/pyout/"torchaudio-*.whl >/dev/null 2>&1; then
        suggestions+=("vision (torchaudio wheel exists)")
    fi
    
    if [ ${#suggestions[@]} -gt 0 ]; then
        echo "Detected possible continue points:"
        for suggestion in "${suggestions[@]}"; do
            echo "  --continue $suggestion"
        done
        echo ""
    fi
}

if [ -z "$CONTINUE_FROM" ]; then
    check_build_state
fi

echo "=== Building PyTorch with AOTriton for gfx1151 ==="
[ -n "$CONTINUE_FROM" ] && echo "Continuing from stage: $CONTINUE_FROM"

# Stage: Environment check
if ! should_skip_to "venv"; then
    show_stage "Environment Check"
    PYTHON_BIN=$(command -v "$PYTHON_EXE")
    echo "Using python interpreter: $PYTHON_BIN"
    "$PYTHON_EXE" --version
    if ! "$PYTHON_EXE" -c "import torch" >/dev/null 2>&1; then
        cat <<'EOF'
Error: Could not import torch with the selected Python interpreter.
Ensure you're running this script inside an environment with ROCm-enabled PyTorch dependencies.
You can run ./00-setup-env.sh to provision a conda environment, or supply your own.
EOF
        exit 1
    fi
    TORCH_DETAILS=$("$PYTHON_EXE" - <<'PY'
import torch, os
loc = os.path.dirname(torch.__file__)
hip = getattr(torch.version, 'hip', 'unknown')
print(f"version={torch.__version__} hip={hip} location={loc}")
PY
)
    echo "Torch details: $TORCH_DETAILS"
else
    echo "SKIPPED: Environment check (assuming prerequisites satisfied)"
fi

# Set up environment for gfx1151 (always needed)
export PYTORCH_ROCM_ARCH=gfx1151
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Enable torch.distributed with Gloo backend (CPU) for import compatibility
# This keeps ROCm compute enabled while avoiding RCCL/NCCL requirements.
export USE_DISTRIBUTED=1
export USE_GLOO=1
export USE_RCCL=0
export USE_NCCL=0
# Avoid attempting IB verbs support when using Gloo-only
export USE_IBVERBS=0

# Disable fbgemm GPU/GenAI (CUDA-only components) for ROCm builds
export USE_FBGEMM_GPU=0
export USE_FBGEMM_GENAI=0

# Disable ROCm SMI (librocm_smi64) linkage to avoid rsmi_* unresolved symbols when
# the SMI library is not present in the SDK/runtime path.
export USE_ROCM_SMI=0

# Also force-disable via CMake to cover differing option names across versions
export CMAKE_ARGS="${CMAKE_ARGS} -DUSE_ROCM_SMI=OFF -DROCM_USE_SMI=OFF -DROCM_ENABLE_SMI=OFF -DROCM_SMI_SUPPORT=OFF -DUSE_RSMI=OFF -DATEN_USE_RSMI=OFF -DATEN_WITH_RSMI=OFF"

# Belt-and-suspenders: force the macros off in compilation too
export CXXFLAGS="${CXXFLAGS} -DUSE_ROCM_SMI=0 -DUSE_RSMI=0 -DATEN_USE_RSMI=0"
export CFLAGS="${CFLAGS} -DUSE_ROCM_SMI=0 -DUSE_RSMI=0 -DATEN_USE_RSMI=0"

# Also explicitly disable FBGEMM GPU/GENAI in CMake cache
export CMAKE_ARGS="${CMAKE_ARGS} -DUSE_FBGEMM_GPU=OFF -DUSE_FBGEMM_GENAI=OFF"

# Disable CUDA since we're building for ROCm
export USE_CUDA=0

# Fix Triton version string to avoid multiple '+' characters
export TRITON_WHEEL_VERSION_SUFFIX="+rocm$(date +%Y%m%d)"

# Prefer prebuilt AOTriton from 01 (aotriton/build/install_dir)
LOCAL_AOTRITON_PREFIX="$SCRIPT_DIR/aotriton/build/install_dir"
if [ -z "$AOTRITON_INSTALLED_PREFIX" ] && [ -d "$LOCAL_AOTRITON_PREFIX" ] && [ -f "$LOCAL_AOTRITON_PREFIX/lib/libaotriton_v2.so" ]; then
    echo "Found local prebuilt AOTriton (01) at: $LOCAL_AOTRITON_PREFIX"
    export AOTRITON_INSTALLED_PREFIX="$LOCAL_AOTRITON_PREFIX"
fi

# Fallback: try to locate via Python module to assist users who installed elsewhere
if [ -z "$AOTRITON_INSTALLED_PREFIX" ]; then
    AOTRITON_SITE_ROOT=$("$PYTHON_EXE" -c "import os,sys;\
try:\n import pyaotriton; p=os.path.dirname(pyaotriton.__file__);\
 # common layout from source install keeps lib/ one level up from module
 cands=[os.path.abspath(os.path.join(p, '..')), os.path.abspath(os.path.join(p, '..', '..'))];\
print('\n'.join(cands))\nexcept Exception:\n pass" 2>/dev/null | head -n1)
    if [ -n "$AOTRITON_SITE_ROOT" ] && [ -d "$AOTRITON_SITE_ROOT/lib" ] && [ -f "$AOTRITON_SITE_ROOT/lib/libaotriton_v2.so" ]; then
        echo "Found AOTriton via Python site at: $AOTRITON_SITE_ROOT"
        export AOTRITON_INSTALLED_PREFIX="$AOTRITON_SITE_ROOT"
    fi
fi

# If still not found, build from source inside PyTorch build
if [ -z "$AOTRITON_INSTALLED_PREFIX" ]; then
    echo "Prebuilt AOTriton not found; will build from source (this may fail on some hosts)."
    export AOTRITON_INSTALL_FROM_SOURCE=1
else
    echo "Using AOTRITON_INSTALLED_PREFIX=$AOTRITON_INSTALLED_PREFIX"
fi

# Navigate to pytorch build directory
cd $PYTORCH_BUILD_DIR

# Stage: Repository Checkout
if ! should_skip_to "checkout"; then
    show_stage "Repository Checkout"
    echo "Checking out PyTorch repositories..."
    if [ ! -d "pytorch" ]; then
        "$PYTHON_EXE" pytorch_torch_repo.py checkout --repo-hashtag main --patchset main
    else
        echo "pytorch directory already exists, skipping checkout"
    fi
    if [ ! -d "pytorch_audio" ]; then
        "$PYTHON_EXE" pytorch_audio_repo.py checkout --repo-hashtag main
    else
        echo "pytorch_audio directory already exists, skipping checkout"
    fi
    if [ ! -d "pytorch_vision" ]; then
        "$PYTHON_EXE" pytorch_vision_repo.py checkout --repo-hashtag main
    else
        echo "pytorch_vision directory already exists, skipping checkout"  
    fi
    
    # Triton requires special handling for patches - remove and recheckout if patches weren't applied
    if [ ! -d "triton" ]; then
        echo "Checking out triton with patches..."
        "$PYTHON_EXE" pytorch_triton_repo.py checkout --patch --patchset nightly
    else
        # Check if the triton patch was applied by looking for the fix function
        if ! grep -q "get_triton_version_suffix" triton/setup.py; then
            echo "Triton directory exists but patches not applied - removing and recheckingout..."
            rm -rf triton
            "$PYTHON_EXE" pytorch_triton_repo.py checkout --patch --patchset nightly
        else
            echo "triton directory already exists with patches applied, skipping checkout"
        fi
    fi
else
    echo "SKIPPED: Repository checkout"
fi

# Patch PyTorch aotriton.cmake to respect AOTRITON_INSTALLED_PREFIX for linking
patch_aotriton_cmake() {
    local cmake_file="$PYTORCH_BUILD_DIR/pytorch/cmake/External/aotriton.cmake"
    if [ ! -f "$cmake_file" ]; then
        echo "AOTriton CMake file not found (skipping patch): $cmake_file"
        return 0
    fi
    # If already patched (looks for installed-prefix-based lib path), skip
    if rg -n 'AOTRITON_LIB\s*\"\$\{__AOTRITON_INSTALL_DIR\}/lib/libaotriton_v2.so\"' "$cmake_file" >/dev/null 2>&1; then
        echo "AOTriton CMake already patched"
        return 0
    fi
    # Insert block after the line that sets __AOTRITON_INSTALL_DIR from env
    local tmp_file
    tmp_file="${cmake_file}.tmp.$$"
    awk '
        BEGIN{inserted=0}
        {
          print $0
          if (!inserted && $0 ~ /set\(__AOTRITON_INSTALL_DIR \"\$ENV\{AOTRITON_INSTALLED_PREFIX\}\"\)/) {
            print "    # Ensure AOTRITON_LIB reflects the chosen install prefix";
            print "    if(NOT WIN32)";
            print "      set(AOTRITON_LIB \"${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.so\")";
            print "    else()";
            print "      set(AOTRITON_LIB \"${__AOTRITON_INSTALL_DIR}/lib/aotriton_v2.lib\")";
            print "      set(AOTRITON_DLL \"${__AOTRITON_INSTALL_DIR}/lib/aotriton_v2.dll\")";
            print "    endif()";
            inserted=1;
          }
        }
        END{ if (!inserted) { exit 1 } }
    ' "$cmake_file" > "$tmp_file" || {
        echo "Failed to apply AOTriton CMake patch automatically; please update manually." >&2
        rm -f "$tmp_file"
        return 1
    }
    mv "$tmp_file" "$cmake_file"
    echo "Patched: $cmake_file"
}

patch_aotriton_cmake

# Patch torchaudio's LoadHIP.cmake to gracefully handle missing version-dev
patch_torchaudio_hip() {
    local hip_cmake_file="$PYTORCH_BUILD_DIR/pytorch_audio/cmake/LoadHIP.cmake"
    if [ ! -f "$hip_cmake_file" ]; then
        echo "Torchaudio LoadHIP.cmake not found (skipping patch): $hip_cmake_file"
        return 0
    fi
    # Skip if already patched
    if grep -q "EXISTS.*version-dev" "$hip_cmake_file"; then
        echo "Torchaudio LoadHIP.cmake already patched"
        return 0
    fi
    
    # Simple fix: wrap file read with EXISTS check and regex with empty check
    sed -i '165s|.*file(READ.*|    if(EXISTS "${ROCM_PATH}/.info/version-dev")\n      file(READ "${ROCM_PATH}/.info/version-dev" ${ROCM_LIB_NAME}_VERSION_DEV_RAW)\n    else()\n      set(${ROCM_LIB_NAME}_VERSION_DEV_RAW "")\n    endif()|' "$hip_cmake_file"
    
    # Also guard the string REGEX operation
    sed -i '/string(REGEX MATCH.*VERSION_DEV_MATCH.*VERSION_DEV_RAW)/i\
  if(DEFINED ${ROCM_LIB_NAME}_VERSION_DEV_RAW AND NOT "${${ROCM_LIB_NAME}_VERSION_DEV_RAW}" STREQUAL "")' "$hip_cmake_file"
    sed -i '/string(REGEX MATCH.*VERSION_DEV_MATCH.*VERSION_DEV_RAW)/a\
  endif()' "$hip_cmake_file"
    
    echo "Patched: $hip_cmake_file (version-dev file and regex checks)"
}

patch_torchaudio_hip

# Report AOTriton pin and prebuilt alignment (advisory)
report_aotriton_pin() {
    local cmake_file="$PYTORCH_BUILD_DIR/pytorch/cmake/External/aotriton.cmake"
    if [ -f "$cmake_file" ]; then
        local cmake_pin
        cmake_pin=$(awk -F '"' '/set\(__AOTRITON_CI_COMMIT/{print $2; exit}' "$cmake_file")
        if [ -n "$cmake_pin" ]; then
            echo "AOTriton pin from PyTorch: $cmake_pin"
        fi
        # If using local prebuilt, compare its HEAD commit
        local local_src_dir="$SCRIPT_DIR/aotriton"
        if [ -d "$local_src_dir/.git" ] && [ "$AOTRITON_INSTALLED_PREFIX" = "$SCRIPT_DIR/aotriton/build/install_dir" ]; then
            local local_head
            local_head=$(git -C "$local_src_dir" rev-parse --short=40 HEAD 2>/dev/null || true)
            if [ -n "$local_head" ]; then
                echo "Local prebuilt AOTriton commit: $local_head"
                if [ -n "$cmake_pin" ] && [ "$cmake_pin" != "$local_head" ]; then
                    echo "WARNING: Prebuilt AOTriton commit does not match PyTorch pin." >&2
                    echo "         Consider rebuilding 01-build-aotriton.sh at $cmake_pin or set AOTRITON_INSTALL_FROM_SOURCE=1." >&2
                fi
            fi
        fi
    else
        echo "AOTriton CMake file not found for pin report: $cmake_file"
    fi
}

report_aotriton_pin

# Patch torchaudio CMake to handle missing kineto library gracefully
patch_torchaudio_kineto() {
    local cmake_file="$PYTORCH_BUILD_DIR/pytorch_audio/cmake/TorchAudioHelper.cmake"
    if [ ! -f "$cmake_file" ]; then
        echo "Torchaudio TorchAudioHelper.cmake not found (skipping patch): $cmake_file"
        return 0
    fi
    
    # Check if already patched 
    if grep -q "Handle missing kineto" "$cmake_file"; then
        echo "Torchaudio kineto CMake already patched"
        return 0
    fi
    
    # Insert kineto handling after the find_package(Torch REQUIRED) line
    sed -i '/find_package(Torch REQUIRED)/a\\n# Handle missing kineto library gracefully\nif(TARGET torch::kineto OR kineto_LIBRARY)\n    message(STATUS "Kineto profiling support enabled")\n    set(USE_KINETO ON)\nelse()\n    message(WARNING "Kineto profiling library not found - profiling features disabled")\n    set(USE_KINETO OFF)\n    # Create empty kineto target to satisfy dependencies\n    add_library(kineto INTERFACE)\n    add_library(torch::kineto ALIAS kineto)\nendif()' "$cmake_file"
    
    echo "Patched: $cmake_file (kineto handling)"
}

patch_torchaudio_kineto

# Patch gloo to include <cstdint> for fixed-width integer types on some toolchains
patch_gloo_cstdint() {
    local header_file="$PYTORCH_BUILD_DIR/pytorch/third_party/gloo/gloo/types.h"
    if [ ! -f "$header_file" ]; then
        echo "Gloo types.h not found (skipping patch): $header_file"
        return 0
    fi
    # Skip if already patched
    if grep -q "#include <cstdint>" "$header_file"; then
        echo "Gloo types.h already includes <cstdint>"
        return 0
    fi
    # Insert <cstdint> include after the common header include
    local tmp_file
    tmp_file="${header_file}.tmp.$$"
    awk '
        BEGIN{inserted=0}
        {
          print $0
          if (!inserted && $0 ~ /#include\s+\"gloo\/common\/common.h\"/) {
            print "#include <cstdint>";
            inserted=1;
          }
        }
        END{ if (!inserted) { exit 1 } }
    ' "$header_file" > "$tmp_file" || {
        echo "Failed to patch gloo/types.h to include <cstdint>; please add it manually." >&2
        rm -f "$tmp_file"
        return 1
    }
    mv "$tmp_file" "$header_file"
    echo "Patched: $header_file (added <cstdint>)"
}

patch_gloo_cstdint

# Patch out any ROCm SMI usages in HIP sources if present
patch_disable_rsmi_code() {
    local root="$PYTORCH_BUILD_DIR/pytorch"
    if [ ! -d "$root" ]; then
        echo "PyTorch source not found for RSMI patch (skipping): $root"
        return 0
    fi
    # Find candidate files referencing ROCm SMI headers or rsmi_* APIs
    mapfile -t files < <(rg -n --no-heading -l "rocm_smi/rocm_smi.h|\\brsmi_" "$root" 2>/dev/null || true)
    if [ ${#files[@]} -eq 0 ]; then
        echo "No ROCm SMI references found to patch"
        return 0
    fi
    echo "Patching RSMI references in:"
    for f in "${files[@]}"; do
        echo "  $f"
        # 1) Comment out direct include of rocm_smi header
        sed -i 's|^[[:space:]]*#include[[:space:]]*[<\"]rocm_smi/rocm_smi.h[>\"]|#if 0\n&\n#endif|' "$f" || true
        # 2) Force any preprocessor checks to false
        sed -i 's/^\([[:space:]]*#if\)[[:space:]]\+defined(\?USE_ROCM_SMI\()?\)/\1 0/' "$f" || true
        sed -i 's/^\([[:space:]]*#ifdef\)[[:space:]]\+USE_ROCM_SMI/\1 DISABLED_USE_ROCM_SMI/' "$f" || true
        sed -i 's/^\([[:space:]]*#if\)[[:space:]]\+USE_ROCM_SMI/\1 0/' "$f" || true
    done
}

patch_disable_rsmi_code

# Guard FBGEMM GenAI includes behind __has_include for ROCm builds
patch_fbgemm_genai_guards() {
    local hip_file="$PYTORCH_BUILD_DIR/pytorch/aten/src/ATen/native/hip/Blas.cpp"
    local cuda_file="$PYTORCH_BUILD_DIR/pytorch/aten/src/ATen/native/cuda/Blas.cpp"

    for f in "$hip_file" "$cuda_file"; do
        if [ ! -f "$f" ]; then
            echo "FBGEMM guard patch: file not found (skipping): $f"
            continue
        fi
        if rg -n "__has_include\(<fbgemm_gpu/torch_ops.h>\)" "$f" >/dev/null 2>&1; then
            echo "FBGEMM guard patch: already patched: $f"
            continue
        fi
        # Replace all occurrences of "#ifdef USE_FBGEMM_GENAI" with a safer check
        sed -E -i 's/^([[:space:]]*)#ifdef USE_FBGEMM_GENAI/\1#if defined(USE_FBGEMM_GENAI) \&\& __has_include(<fbgemm_gpu\/torch_ops.h>)/' "$f" || {
            echo "Failed to patch FBGEMM guard in: $f" >&2
            continue
        }
        echo "Patched FBGEMM GenAI guard: $f"
    done
}

patch_fbgemm_genai_guards

# Force-disable FBGEMM_GENAI in build_prod_wheels.py for Linux (<2.10 path)
patch_build_prod_wheels_disable_fbgemm() {
    local bpw_file="$THEROCK_DIR/external-builds/pytorch/build_prod_wheels.py"
    if [ ! -f "$bpw_file" ]; then
        echo "build_prod_wheels.py not found (skipping FBGEMM disable patch): $bpw_file"
        return 0
    fi
    # If this line already uses OFF, skip. Otherwise, switch ON->OFF in the <2.10 branch
    if rg -n "FBGEMM_GENAI pre-set \(honored\)" "$bpw_file" >/dev/null 2>&1; then
        echo "FBGEMM disable patch: already modernized (honors env)"
        return 0
    fi
    if rg -n "env\[\"USE_FBGEMM_GENAI\"\] = \"OFF\"" "$bpw_file" >/dev/null 2>&1; then
        echo "FBGEMM disable patch: already OFF by default"
        return 0
    fi
    sed -i '0,/env\["USE_FBGEMM_GENAI"\] = "ON"/s//env["USE_FBGEMM_GENAI"] = "OFF"/' "$bpw_file" && \
        echo "Patched build_prod_wheels to default USE_FBGEMM_GENAI=OFF for <2.10"
}

patch_build_prod_wheels_disable_fbgemm

# Speed up and avoid linking tests that can fail when optional deps are off
export BUILD_TEST=0

# Patch intra_node_comm.cpp to hard-disable any RSMI usage during build
patch_intra_node_comm() {
    local f="$PYTORCH_BUILD_DIR/pytorch/torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cpp"
    if [ ! -f "$f" ]; then
        echo "intra_node_comm.cpp not found (skipping RSMI guard patch): $f"
        return 0
    fi
    # 1) Force non-RSMI path in getNvlMesh by avoiding USE_RCOM branch
    if rg -n "#if !defined\(USE_RCOM\)" "$f" >/dev/null 2>&1; then
        sed -i 's/#if !defined(USE_RCOM)/#if 1/' "$f"
        echo "Patched getNvlMesh guard to avoid RSMI path"
    fi
    # 2) Disable RSMI init in rendezvous by turning the nearby guard to #if 0 (second occurrence)
    local tmp="${f}.tmp.$$"
    awk '
        BEGIN{count_if=0}
        {
            if ($0 ~ /^#if defined\(USE_ROCM\)/) { count_if++; }
            if (count_if==2 && $0 ~ /^#if defined\(USE_ROCM\)/) { print "#if 0"; next }
            print $0
        }
    ' "$f" > "$tmp" && mv "$tmp" "$f" && echo "Disabled RSMI init block in rendezvous"
}

patch_intra_node_comm

# Stage: Build with conditional stages
BUILD_ARGS=(
    --pytorch-rocm-arch gfx1151
    --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/
    --output-dir $HOME/tmp/pyout
    --clean
)

# Add ROCm installation if needed
if ! should_skip_to "rocm"; then
    BUILD_ARGS+=(--install-rocm)
else
    echo "SKIPPED: ROCm installation"
fi

# Add triton build control
if should_skip_to "triton"; then
    echo "SKIPPED: Triton build"
    BUILD_ARGS+=(--no-build-triton)
fi

# Add pytorch build control  
if should_skip_to "pytorch"; then
    echo "SKIPPED: PyTorch build"
    # Create a dummy pytorch dir to fool the build system if needed
    mkdir -p pytorch_dummy
    BUILD_ARGS+=(--pytorch-dir pytorch_dummy)
fi

# Add audio build control
if should_skip_to "audio"; then
    echo "SKIPPED: Torchaudio build" 
    BUILD_ARGS+=(--no-build-pytorch-audio)
fi

# Add vision build control
if should_skip_to "vision"; then
    echo "SKIPPED: Torchvision build"
    BUILD_ARGS+=(--no-build-pytorch-vision)  
fi

show_stage "Build PyTorch with AOTriton for gfx1151"
echo "Build command: \"$PYTHON_EXE\" build_prod_wheels.py build ${BUILD_ARGS[*]}"
# Honor skip flags regardless of --continue
if [ "$SKIP_AUDIO" = "1" ]; then
    BUILD_ARGS+=(--no-build-pytorch-audio)
    echo "Flag: --skip-audio enabled (torchaudio will be skipped)"
fi
if [ "$SKIP_VISION" = "1" ]; then
    BUILD_ARGS+=(--no-build-pytorch-vision)
    echo "Flag: --skip-vision enabled (torchvision will be skipped)"
fi

"$PYTHON_EXE" build_prod_wheels.py build "${BUILD_ARGS[@]}"

show_stage "Build Complete"
echo "Built wheels are in: $HOME/tmp/pyout"
echo ""
echo "To install (exact wheels for this run):"
PYTAG=$("$PYTHON_EXE" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
RUNSTAMP=$(date +%Y%m%d)
OUTDIR="$HOME/tmp/pyout"

# Prefer wheels built today for the active Python tag
TORCH_WHL=$(ls -1t "$OUTDIR"/torch-*+rocmsdk${RUNSTAMP}-${PYTAG}-${PYTAG}-*.whl 2>/dev/null | head -n1)
AUDIO_WHL=$(ls -1t "$OUTDIR"/torchaudio-*+rocmsdk${RUNSTAMP}-${PYTAG}-${PYTAG}-*.whl 2>/dev/null | head -n1)
VISION_WHL=$(ls -1t "$OUTDIR"/torchvision-*+rocmsdk${RUNSTAMP}-${PYTAG}-${PYTAG}-*.whl 2>/dev/null | head -n1)

if [ -n "$TORCH_WHL" ]; then
  echo "  pip install \"$TORCH_WHL\" --force-reinstall --no-deps"
else
  echo "  (torch) No exact-match wheel found for ${PYTAG} + rocmsdk${RUNSTAMP}. Check $OUTDIR."
fi
if [ -n "$AUDIO_WHL" ]; then
  echo "  pip install \"$AUDIO_WHL\" --force-reinstall --no-deps"
fi
if [ -n "$VISION_WHL" ]; then
  echo "  pip install \"$VISION_WHL\" --force-reinstall --no-deps"
fi
