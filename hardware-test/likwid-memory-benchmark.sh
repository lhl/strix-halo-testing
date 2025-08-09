#!/bin/bash

# Memory Bandwidth Benchmark Script using likwid
# Automatically installs likwid if needed and runs comprehensive memory tests

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_FILE="likwid-memory-benchmark-$(date +%Y%m%d_%H%M%S).txt"
WORKING_SET="S0:1GB"
TEST_DURATION="10s"
ITERATIONS=3

echo -e "${BLUE}=== Memory Bandwidth Benchmark Script ===${NC}"
echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$OUTPUT_FILE"
}

# Check if likwid is installed
if ! command_exists likwid-bench; then
    echo -e "${YELLOW}likwid not found. Installing...${NC}"
    if command_exists pacman; then
        sudo pacman -S --noconfirm likwid
        echo -e "${GREEN}likwid installed successfully!${NC}"
    else
        echo -e "${RED}Error: pacman not found. This script is for Arch Linux.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}likwid is already installed.${NC}"
fi

# Verify installation
if ! command_exists likwid-bench; then
    echo -e "${RED}Error: likwid-bench still not available after installation.${NC}"
    exit 1
fi

echo ""
log_message "=== System Information ==="
log_message "Hostname: $(hostname)"
log_message "CPU: $(lscpu | grep 'Model name' | sed 's/Model name:[[:space:]]*//')"
log_message "Memory: $(free -h | grep 'Mem:' | awk '{print $2}')"
log_message "Kernel: $(uname -r)"
log_message ""

# Function to run benchmark and log
run_benchmark() {
    local test_name="$1"
    local benchmark_type="$2"
    local extra_args="$3"
    
    echo -e "${BLUE}Running $test_name...${NC}"
    log_message "=== $test_name ==="
    
    if likwid-bench -t "$benchmark_type" -w "$WORKING_SET" -s "$TEST_DURATION" -i "$ITERATIONS" $extra_args 2>&1 | tee -a "$OUTPUT_FILE"; then
        echo -e "${GREEN}✓ $test_name completed${NC}"
    else
        echo -e "${RED}✗ $test_name failed${NC}" | tee -a "$OUTPUT_FILE"
    fi
    log_message ""
    echo ""
}

# List available benchmarks first
echo -e "${BLUE}Available benchmark kernels:${NC}"
likwid-bench -a | tee -a "$OUTPUT_FILE"
echo ""

# Run comprehensive memory bandwidth tests
log_message "=== Starting Memory Bandwidth Benchmarks ==="
log_message "Working Set: $WORKING_SET"
log_message "Test Duration: $TEST_DURATION"
log_message "Iterations: $ITERATIONS"
log_message ""

# Core memory bandwidth tests
run_benchmark "Copy Benchmark" "copy"
run_benchmark "STREAM Benchmark" "stream" 
run_benchmark "Triad Benchmark (a = b + c*d)" "triad"
run_benchmark "Load Benchmark" "load"
run_benchmark "Store Benchmark" "store"

# Additional tests if available
echo -e "${BLUE}Running additional benchmarks...${NC}"
run_benchmark "Update Benchmark" "update"
run_benchmark "Scale Benchmark" "scale"

# Multi-threaded test
echo -e "${BLUE}Running multi-threaded STREAM test...${NC}"
log_message "=== Multi-threaded STREAM Test ==="
if likwid-bench -t stream -w "$WORKING_SET" -s "$TEST_DURATION" -i "$ITERATIONS" 2>&1 | tee -a "$OUTPUT_FILE"; then
    echo -e "${GREEN}✓ Multi-threaded test completed${NC}"
else
    echo -e "${RED}✗ Multi-threaded test failed${NC}" | tee -a "$OUTPUT_FILE"
fi

# Summary
log_message "=== Benchmark Summary ==="
log_message "All benchmarks completed at $(date)"
log_message "Results saved to: $OUTPUT_FILE"

echo ""
echo -e "${GREEN}=== Benchmark Complete! ===${NC}"
echo -e "${BLUE}Results saved to: $OUTPUT_FILE${NC}"
echo ""
echo -e "${YELLOW}Quick summary - look for these key metrics in the output:${NC}"
echo "• Bandwidth [MBytes/s] - Higher is better"
echo "• Cycles per element - Lower is better"
echo "• Runtime [s] - Actual test duration"
echo ""
echo -e "${BLUE}To view results: cat $OUTPUT_FILE${NC}"
echo -e "${BLUE}To view just bandwidth results: grep -A2 'Bandwidth' $OUTPUT_FILE${NC}"
