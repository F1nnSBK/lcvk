#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Define ANSI color codes for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Starting Dockerized Vector DB Compilation ===${NC}"

# 1. Build the builder stage which compiles Java code and runs AOT native-image compilation
echo -e "${BLUE}[1/3] Building docker image (target: builder)...${NC}"
docker build -t vectordb-builder --target builder .

# 2. Spin up a temporary container to extract target artifacts
echo -e "${BLUE}[2/3] Extracting compilation artifacts (.so and .h) from container...${NC}"
CONTAINER_ID=$(docker create vectordb-builder)

# Create a host directory for the build output
mkdir -p build-output

# Copy files from docker container to host
docker cp "${CONTAINER_ID}:/build/target/lunar_core.so" ./build-output/liblunar_core.so
docker cp "${CONTAINER_ID}:/build/target/lunar_core.h" ./build-output/
docker cp "${CONTAINER_ID}:/build/target/graal_isolate.h" ./build-output/

# Clean up the container
docker rm "${CONTAINER_ID}"

# 3. Done! Display generated files
echo -e "${GREEN}[3/3] Success! Library and C headers exported to './build-output/' directory:${NC}"
ls -lh ./build-output/

echo -e "\n${GREEN}To compile and test the library locally on Linux (or inside Docker):${NC}"
echo "See verify.sh to run the C test client inside the Docker container."
