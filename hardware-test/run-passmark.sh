#!/bin/bash

# PassMark PerformanceTest Benchmark Script
# Automatically installs passmark-performancetest-bin if needed and runs comprehensive tests

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PACKAGE_NAME="passmark-performancetest-bin"
OUTPUT_FILE="passmark_benchmark_$(date +%Y%m%d_%H%M%S).txt"
POSSIBLE_EXECUTABLES=("pt_linux_x64" "passmark-performancetest" "performancetest")

echo -e "${BLUE}=== PassMark PerformanceTest Benchmark Script ===${NC}"
echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to find the correct executable
find_passmark_executable() {
    for exe in "${POSSIBLE_EXECUTABLES[@]}"; do
        if command_exists "$exe"; then
            echo "$exe"
            return 0
        fi
    done
    return 1
}

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$OUTPUT_FILE"
}

# Check if PassMark is installed
PASSMARK_EXE=""
if PASSMARK_EXE=$(find_passmark_executable); then
    echo -e "${GREEN}PassMark PerformanceTest found: $PASSMARK_EXE${NC}"
else
    echo -e "${YELLOW}PassMark PerformanceTest not found. Installing via paru...${NC}"
    
    # Check if paru is installed
    if ! command_exists paru; then
        echo -e "${RED}Error: paru not found. Please install paru first:${NC}"
        echo "sudo pacman -S --needed base-devel git"
        echo "git clone https://aur.archlinux.org/paru.git"
        echo "cd paru && makepkg -si"
        exit 1
    fi
    
    # Install PassMark PerformanceTest
    echo "Installing $PACKAGE_NAME..."
    if paru -S --noconfirm "$PACKAGE_NAME"; then
        echo -e "${GREEN}$PACKAGE_NAME installed successfully!${NC}"
    else
        echo -e "${RED}Error: Failed to install $PACKAGE_NAME${NC}"
        echo "You may need to download PassMark PerformanceTest manually from:"
        echo "https://www.passmark.com/products/pt_linux/download.php"
        exit 1
    fi
    
    # Try to find the executable again
    if PASSMARK_EXE=$(find_passmark_executable); then
        echo -e "${GREEN}PassMark executable found: $PASSMARK_EXE${NC}"
    else
        echo -e "${RED}Error: PassMark executable still not found after installation.${NC}"
        echo "The package may have installed the executable with a different name."
        echo "Please check: whereis pt_linux_x64 or locate performancetest"
        exit 1
    fi
fi

echo ""

# System information
log_message "=== System Information ==="
log_message "Hostname: $(hostname)"
log_message "CPU: $(lscpu | grep 'Model name' | sed 's/Model name:[[:space:]]*//' || echo 'Unknown')"
log_message "Memory: $(free -h | grep 'Mem:' | awk '{print $2}' || echo 'Unknown')"
log_message "Kernel: $(uname -r)"
log_message "PassMark Executable: $PASSMARK_EXE"
log_message ""

# Check if we need to set library paths (for manual installations)
if [[ "$PASSMARK_EXE" == "./pt_linux_x64" ]]; then
    echo -e "${YELLOW}Note: Using local executable. Checking dependencies...${NC}"
    # Check for common missing libraries
    if ! ldd "$PASSMARK_EXE" >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Some dependencies may be missing.${NC}"
        echo "You may need to install: sudo pacman -S ncurses5-compat-libs"
    fi
fi

# Run PassMark PerformanceTest
echo -e "${BLUE}Starting PassMark PerformanceTest...${NC}"
echo -e "${BLUE}Running: $PASSMARK_EXE -r 3 -d 2 -i 3 -debug${NC}"
echo ""

log_message "=== PassMark PerformanceTest Benchmark ==="
log_message "Command: $PASSMARK_EXE -r 3 -d 2 -i 3 -debug"
log_message "Test Duration: Medium (-d 2)"
log_message "Iterations: 3 (-i 3)"
log_message "Test Suite: All tests (-r 3)"
log_message "Debug logging: Enabled"
log_message ""

# Create a function to handle the benchmark execution
run_passmark() {
    local start_time=$(date)
    log_message "Benchmark started at: $start_time"
    
    # Run PassMark and capture both stdout and stderr
    if "$PASSMARK_EXE" -r 3 -d 2 -i 3 -debug 2>&1 | tee -a "$OUTPUT_FILE"; then
        local end_time=$(date)
        log_message ""
        log_message "Benchmark completed successfully at: $end_time"
        echo -e "${GREEN}✓ PassMark benchmark completed successfully!${NC}"
    else
        local end_time=$(date)
        log_message ""
        log_message "Benchmark failed at: $end_time"
        echo -e "${RED}✗ PassMark benchmark failed!${NC}"
        echo "Check the log file for details: $OUTPUT_FILE"
        return 1
    fi
}

# Execute the benchmark
run_passmark

echo ""
log_message "=== Results Summary ==="

# Check for result files
RESULT_FILES=("results_all.yml" "results_cpu.yml" "results_memory.yml" "debug.log")
for file in "${RESULT_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        log_message "Generated: $file"
        echo -e "${GREEN}✓ Found: $file${NC}"
    fi
done

log_message ""
log_message "All outputs saved to: $OUTPUT_FILE"

echo ""
echo -e "${GREEN}=== PassMark Benchmark Complete! ===${NC}"
echo -e "${BLUE}Results saved to: $OUTPUT_FILE${NC}"
echo ""

# Display result file locations
echo -e "${YELLOW}Result files to check:${NC}"
for file in "${RESULT_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "• $file"
    fi
done

echo ""
echo -e "${YELLOW}Quick commands to view results:${NC}"
echo "• View YAML results: cat results_all.yml"
echo "• View debug log: cat debug.log"
echo "• View benchmark log: cat $OUTPUT_FILE"
echo "• Search for scores: grep -i 'mark\\|score\\|rating' results_all.yml"

# Optional: Display a quick summary if results_all.yml exists
if [[ -f "results_all.yml" ]]; then
    echo ""
    echo -e "${BLUE}Quick Results Preview:${NC}"
    if command_exists grep && command_exists head; then
        grep -E "(CPU|Memory|PassMark)" results_all.yml | head -10 || echo "Could not parse YAML results"
    fi
fi
