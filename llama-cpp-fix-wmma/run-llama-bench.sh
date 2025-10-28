#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIP_BIN="${ROOT_DIR}/llama.cpp-hip/build/bin/llama-bench"
ROCWMMA_BIN="${ROOT_DIR}/llama.cpp-rocwmma/build/bin/llama-bench"

MODEL_PATH=""
TAG=""
DEPTHS="${DEPTHS:-0,1024,4096,16384,65536}"
HIP_OUTPUT=""
ROCWMMA_OUTPUT=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --model PATH --tag NAME [--depths LIST] [--hip-output FILE] [--rocwmma-output FILE]

Environment overrides:
  DEPTHS           Comma-separated context depths (default: ${DEPTHS})
  HIP_OUTPUT       Output file for the HIP build (default: <tag>.default-hip.jsonl)
  ROCWMMA_OUTPUT   Output file for the rocWMMA build (default: <tag>.default-rocwmma.jsonl)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hip-output)
      HIP_OUTPUT="$2"
      shift 2
      ;;
    --rocwmma-output)
      ROCWMMA_OUTPUT="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --depths)
      DEPTHS="$2"
      shift 2
      ;;
    -*)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" || -z "${TAG}" ]]; then
  usage
fi

if [[ -z "${HIP_OUTPUT}" ]]; then
  HIP_OUTPUT="${TAG}.default-hip.jsonl"
fi
if [[ -z "${ROCWMMA_OUTPUT}" ]]; then
  ROCWMMA_OUTPUT="${TAG}.default-rocwmma.jsonl"
fi

if [[ ! -x "${HIP_BIN}" ]]; then
  echo "error: HIP llama-bench not found at ${HIP_BIN}" >&2
  exit 1
fi

if [[ ! -x "${ROCWMMA_BIN}" ]]; then
  echo "error: rocWMMA llama-bench not found at ${ROCWMMA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "error: model file not found at ${MODEL_PATH}" >&2
  exit 1
fi

COMMON_ARGS=(
  -r 1
  -fa 1
  -d "${DEPTHS}"
  --mmap 0
  -m "${MODEL_PATH}"
  -o jsonl
)

format_duration() {
  local total_seconds=$1
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  if ((hours > 0)); then
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
  else
    printf "%02d:%02d" "$minutes" "$seconds"
  fi
}

run_bench() {
  local label=$1
  local bin=$2
  local output=$3

  echo "Running ${label} benchmark → ${output}"
  local start
  start=$(date +%s)
  "${bin}" "${COMMON_ARGS[@]}" > "${output}"
  local end
  end=$(date +%s)
  local elapsed=$((end - start))
  echo "Finished ${label} in $(format_duration "${elapsed}")"
}

overall_start=$(date +%s)

run_bench "rocWMMA" "${ROCWMMA_BIN}" "${ROCWMMA_OUTPUT}"
run_bench "HIP" "${HIP_BIN}" "${HIP_OUTPUT}"

overall_end=$(date +%s)
overall_elapsed=$((overall_end - overall_start))

echo "Done in $(format_duration "${overall_elapsed}"). Results:"
printf '  %s\n' "${ROCWMMA_OUTPUT}" "${HIP_OUTPUT}"
