#!/bin/bash
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Building C Client Verification Container ===${NC}"
# This builds the final stage of the Dockerfile, compiling the C client and running it
docker build -t vectordb-verifier .

echo -e "\n${BLUE}=== Running Verification Test Client inside Container ===${NC}"
docker run --rm vectordb-verifier

echo -e "\n${GREEN}=== Verification Complete! ===${NC}"
