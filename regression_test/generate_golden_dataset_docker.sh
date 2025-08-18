#!/bin/bash

# Script to generate golden dataset using Docker on M1 Mac
# Uses the pre-built image from GitHub Container Registry
# Run this script from the regression_test directory

set -e  # Exit on any error

# Configuration
DOCKER_IMAGE_NAME="ghcr.io/opera-adt/dist-s1:latest"
CONTAINER_WORK_DIR="/home/ops/dist-s1-data"

echo "Pulling latest Docker image from GitHub Container Registry..."
docker pull "${DOCKER_IMAGE_NAME}"

echo "Docker image pulled successfully: ${DOCKER_IMAGE_NAME}"

# Check if ~/.netrc exists
if [ ! -f ~/.netrc ]; then
    echo "Error: ~/.netrc file not found. Please create it with your earthdata credentials:"
    echo "machine urs.earthdata.nasa.gov"
    echo "    login <username>"
    echo "    password <password>"
    exit 1
fi

echo "Running golden dataset generation in Docker container..."
echo "Working directory: $(pwd)"
echo "Container work directory: ${CONTAINER_WORK_DIR}"

# Run the container with:
# - Current directory (regression_test) mounted to container work directory
# - ~/.netrc file mounted for authentication
# - Interactive terminal
# - Remove container after completion
# - Override entrypoint to run python script directly
# - Platform specification for M1 Mac compatibility
docker run -ti --rm \
    --platform linux/amd64 \
    -v "$(pwd)":"${CONTAINER_WORK_DIR}" \
    -v ~/.netrc:/home/ops/.netrc:ro \
    --entrypoint "/bin/bash" \
    "${DOCKER_IMAGE_NAME}" \
    -l -c "cd ${CONTAINER_WORK_DIR} && python 0_generate_golden_dataset.py"

echo "Golden dataset generation completed!"
echo "Check the current directory for outputs:"
echo "- product_0/ (initial product)"
echo "- golden_dataset/ (final golden dataset)"
echo "- out_0/ and out_1/ (intermediate processing outputs)"