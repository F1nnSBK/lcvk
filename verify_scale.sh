#!/bin/bash
# Automation script to compile the shared library and run the planetary scale benchmark.
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== LCVK: Starting High-Performance Compilation ===${NC}"
# Run standard dockerized compilation (Maven & GraalVM native-image)
./build.sh

echo -e "\n${BLUE}=== LCVK: Building Docker Testing Environment ===${NC}"
# Rebuild the testing container with python3, numpy, and benchmark.py included
docker build -t vectordb-verifier .

echo -e "\n${BLUE}=== LCVK: Running Scale Benchmark inside Docker ===${NC}"
# Run the Python benchmark script inside the Linux container to avoid macOS Mach-O vs Linux ELF conflicts
docker run --rm vectordb-verifier python3 benchmark.py

echo -e "\n${GREEN}=== LCVK: Scale Stress-Test Completed Successfully ===${NC}"
echo -e "Note: Running inside Docker virtualization has CPU-sharing limits on macOS."
echo -e "To get absolute bare-metal performance, run LCVK natively on a Linux host:"
echo -e "  1. Install Java 25 & GraalVM Native Image on host"
echo -e "  2. Compile: mvn clean package"
echo -e "  3. Run benchmark: python3 benchmark.py"
