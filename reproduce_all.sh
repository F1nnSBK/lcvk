#!/bin/bash
set -e

# Define ANSI color codes for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Full Reproducibility Script for Paper ===${NC}"

# 1. Compile native libraries if they don't exist
if [ ! -f "libpithos.dylib" ] && [ ! -f "target/libpithos.dylib" ] && [ ! -f "build-output/libpithos.so" ] && [ ! -f "target/libpithos.so" ]; then
    echo -e "${YELLOW}Native shared library not found. Attempting to build with Maven...${NC}"
    mvn clean package
fi

# 2. Check Python environment
if command -v python3 &> /dev/null && python3 -c "import numpy, matplotlib, faiss" &> /dev/null; then
    PYTHON_BIN="python3"
elif [ -f ".venv/bin/python" ]; then
    echo -e "${YELLOW}Using .venv/bin/python...${NC}"
    PYTHON_BIN=".venv/bin/python"
else
    echo -e "${YELLOW}Using fallback python3...${NC}"
    PYTHON_BIN="python3"
fi

echo -e "\n${BLUE}=== Phase 1: Real-World Data Ingestion & Setup ===${NC}"
PYTHONPATH=. $PYTHON_BIN benchmarks/ingest_pipeline.py

echo -e "\n${BLUE}=== Phase 2: Generating Queries ===${NC}"
PYTHONPATH=. $PYTHON_BIN benchmarks/query_generator.py

echo -e "\n${BLUE}=== Phase 3: Real Data Verification & Evaluation ===${NC}"
PYTHONPATH=. $PYTHON_BIN benchmarks/run_real_verification.py

echo -e "\n${BLUE}=== Phase 4: Running Full Benchmarks & Baselines ===${NC}"
bash run_benchmark.sh

echo -e "\n${GREEN}=== Reproducibility Run Complete! All results, plots and metadata updated. ===${NC}"
