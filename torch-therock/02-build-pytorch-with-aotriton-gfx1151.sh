#!/bin/bash

# Build PyTorch with AOTriton support for gfx1151
# This script uses your prebuilt aotriton installation
#
# Usage:
#   ./02-build-pytorch-with-aotriton-gfx1151.sh [--continue STAGE] [--help]
#
# Stages:
#   checkout   - Skip to repository checkout (after venv setup)
#   rocm       - Skip to ROCm installation 
#   triton     - Skip to triton build
#   pytorch    - Skip to PyTorch build
#   audio      - Skip to torchaudio build
#   vision     - Skip to torchvision build

set -e
set -o pipefail

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
            echo "  checkout   - Skip to repository checkout (after venv setup)"
            echo "  rocm       - Skip to ROCm installation"
            echo "  triton     - Skip to triton build"
            echo "  pytorch    - Skip to PyTorch build"  
            echo "  audio      - Skip to torchaudio build"
            echo "  vision     - Skip to torchvision build"
            echo ""
            echo "Optional flags:"
            echo "  --skip-audio    Skip building torchaudio"
            echo "  --skip-vision   Skip building torchvision"
            echo "  --python-exe    Python interpreter to create venv (e.g., python3.12)"
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
    if pip show rocm-sdk >/dev/null 2>&1; then
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

# Stage: Virtual Environment Setup
if ! should_skip_to "venv"; then
    show_stage "Virtual Environment Setup"
    if [ -n "${VIRTUAL_ENV}" ] || [ -n "${CONDA_PREFIX}" ]; then
        if [ -n "${VIRTUAL_ENV}" ]; then
            echo "Using existing virtual environment: ${VIRTUAL_ENV}"
        else
            echo "Using existing conda environment: ${CONDA_PREFIX}"
        fi
    else
        if [ -f $THEROCK_DIR/.venv/bin/activate ]; then
            source $THEROCK_DIR/.venv/bin/activate
            echo "Activated virtual environment: ${VIRTUAL_ENV}"
        else
            echo "Creating new virtual environment..."
            cd $THEROCK_DIR
            "$PYTHON_EXE" -m venv .venv && source .venv/bin/activate
            echo "Created and activated virtual environment: ${VIRTUAL_ENV}"
        fi
    fi
else
    echo "SKIPPED: Virtual environment setup"
    # Still need to activate if available
    if [ -f $THEROCK_DIR/.venv/bin/activate ]; then
        source $THEROCK_DIR/.venv/bin/activate
    fi
fi

# Set up environment for gfx1151 (always needed)
export PYTORCH_ROCM_ARCH=gfx1151
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Disable distributed training for single GPU builds (avoids NCCL/RCCL issues)
export USE_DISTRIBUTED=0
export USE_NCCL=0
export USE_RCCL=0

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
    AOTRITON_SITE_ROOT=$(python -c "import os,sys;\
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
        python pytorch_torch_repo.py checkout --repo-hashtag main --patchset main
    else
        echo "pytorch directory already exists, skipping checkout"
    fi
    if [ ! -d "pytorch_audio" ]; then
        python pytorch_audio_repo.py checkout --repo-hashtag main
    else
        echo "pytorch_audio directory already exists, skipping checkout"
    fi
    if [ ! -d "pytorch_vision" ]; then
        python pytorch_vision_repo.py checkout --repo-hashtag main
    else
        echo "pytorch_vision directory already exists, skipping checkout"  
    fi
    
    # Triton requires special handling for patches - remove and recheckout if patches weren't applied
    if [ ! -d "triton" ]; then
        echo "Checking out triton with patches..."
        python pytorch_triton_repo.py checkout --patch --patchset nightly
    else
        # Check if the triton patch was applied by looking for the fix function
        if ! grep -q "get_triton_version_suffix" triton/setup.py; then
            echo "Triton directory exists but patches not applied - removing and recheckingout..."
            rm -rf triton
            python pytorch_triton_repo.py checkout --patch --patchset nightly
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
echo "Build command: python build_prod_wheels.py build ${BUILD_ARGS[*]}"
# Honor skip flags regardless of --continue
if [ "$SKIP_AUDIO" = "1" ]; then
    BUILD_ARGS+=(--no-build-pytorch-audio)
    echo "Flag: --skip-audio enabled (torchaudio will be skipped)"
fi
if [ "$SKIP_VISION" = "1" ]; then
    BUILD_ARGS+=(--no-build-pytorch-vision)
    echo "Flag: --skip-vision enabled (torchvision will be skipped)"
fi

python build_prod_wheels.py build "${BUILD_ARGS[@]}"

show_stage "Build Complete"
echo "Built wheels are in: $HOME/tmp/pyout"
echo ""
echo "To install:"
echo "  pip install $HOME/tmp/pyout/torch-*.whl --force-reinstall --no-deps"
echo "  pip install $HOME/tmp/pyout/torchaudio-*.whl --force-reinstall --no-deps"  
echo "  pip install $HOME/tmp/pyout/torchvision-*.whl --force-reinstall --no-deps"
