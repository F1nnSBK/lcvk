#!/bin/bash
set -e

# Define ANSI color codes for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Pithos Open Source Repository Benchmark Runner ===${NC}"
echo -e "${BLUE}(Note: Pithos was formerly known as 'lcvk')${NC}\n"

# 1. Verify GraalVM library is compiled
if [ ! -f "libpithos.dylib" ] && [ ! -f "target/libpithos.dylib" ] && [ ! -f "build-output/libpithos.so" ] && [ ! -f "target/libpithos.so" ]; then
    echo -e "${YELLOW}[Warning] Native shared library not found. Attempting to build with Maven...${NC}"
    if command -v mvn &> /dev/null; then
        mvn clean package
    else
        echo -e "\033[0;31m[Error] Maven is not installed. Please compile libpithos before running this benchmark.\033[0m"
        exit 1
    fi
fi

# 2. Check Python environment
if command -v python3 &> /dev/null && python3 -c "import numpy, matplotlib, faiss" &> /dev/null; then
    PYTHON_BIN="python3"
elif [ -f ".venv/bin/python" ]; then
    echo -e "${YELLOW}[Warning] System python3 does not meet requirements. Using .venv/bin/python...${NC}"
    PYTHON_BIN=".venv/bin/python"
else
    echo -e "${YELLOW}[Warning] Using fallback python3...${NC}"
    PYTHON_BIN="python3"
fi

# Discipline 1: General-Purpose KNN Search
echo -e "${BLUE}=== Step 1/8: Baseline Benchmarks (JIT loop & FAISS) ===${NC}"
$PYTHON_BIN benchmark_baselines.py

echo -e "\n${BLUE}=== Step 2/8: Pithos Scale Performance Benchmark ===${NC}"
$PYTHON_BIN benchmark.py

echo -e "\n${BLUE}=== Step 3/8: Recall@K -- Speed-Accuracy Trade-Off ===${NC}"
$PYTHON_BIN benchmark_recall.py

# Discipline 2: Application-Specific Resonant Voting
echo -e "\n${BLUE}=== Step 4/8: Resonant Voting Stress-Test (Pithos vs FAISS Emulated) ===${NC}"
$PYTHON_BIN benchmark_voting.py

# Dimensionality Crossover Sweep
echo -e "\n${BLUE}=== Step 5/8: Crossover Sweep (Single vs Multi Query x Dimensions) ===${NC}"
$PYTHON_BIN benchmark_sweep.py

# SIFT10K Generalization Benchmark
echo -e "\n${BLUE}=== Step 6/8: SIFT10K Generalization Benchmark ===${NC}"
$PYTHON_BIN benchmark_sift.py

# FFI Boundary Analysis
echo -e "\n${BLUE}=== Step 7/8: FFI Boundary Analysis ===${NC}"
$PYTHON_BIN benchmark_ffi.py

# Candidate Generation Recall (Workload Reduction Elbow Curve)
echo -e "\n${BLUE}=== Step 8/8: Downstream Workload Reduction & Recall Elbow Curve ===${NC}"
$PYTHON_BIN benchmark_candidate_recall.py

# Finalize: regenerate all plots and update README.md
echo -e "\n${BLUE}=== Updating README.md with live benchmark results ===${NC}"
$PYTHON_BIN generate_graphics.py

echo -e "\n${GREEN}=== One-Click Benchmark Complete! ===${NC}"
