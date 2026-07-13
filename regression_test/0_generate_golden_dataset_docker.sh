# Script to generate golden dataset using Docker on M1 Mac
# Uses the pre-built image from GitHub Container Registry
# Run this script from the regression_test directory

set -e  # Exit on any error

# Load DIST_S1_VERSION (and any other vars) from .env in this directory
set -a; [ -f .env ] && source .env; set +a
: "${DIST_S1_VERSION:?Set DIST_S1_VERSION in regression_test/.env (e.g. DIST_S1_VERSION=2.0.18)}"

# Platform for the amd64-only image: default works on amd64 linux (no-op) and
# arm64 Mac (emulation). Override via DOCKER_PLATFORM for other hosts.
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

# Configuration
DOCKER_IMAGE_NAME="ghcr.io/opera-adt/dist-s1:${DIST_S1_VERSION}"
CONTAINER_WORK_DIR="/home/ops/dist-s1-data"

echo "Pulling Docker image from GitHub Container Registry..."
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

# Extract credentials from .netrc as backup
EARTHDATA_USERNAME=$(grep -A2 "machine urs.earthdata.nasa.gov" ~/.netrc | grep "login" | awk '{print $2}')
EARTHDATA_PASSWORD=$(grep -A2 "machine urs.earthdata.nasa.gov" ~/.netrc | grep "password" | awk '{print $2}')

if [ -z "$EARTHDATA_USERNAME" ] || [ -z "$EARTHDATA_PASSWORD" ]; then
    echo "Warning: Could not extract credentials from .netrc. Make sure it's properly formatted."
fi

# Clean up any existing output directories to avoid permission issues
echo "Cleaning up existing output directories..."
rm -rf product_0 golden_dataset out_0 out_1

echo "Running golden dataset generation in Docker container..."
echo "Working directory: $(pwd)"
echo "Container work directory: ${CONTAINER_WORK_DIR}"

# Run the container with:
# - Current directory (regression_test) mounted to container work directory
# - ~/.netrc file mounted for authentication
# - Interactive terminal
# - Remove container after completion
# - Override entrypoint to run python script directly
# - Platform specification (DOCKER_PLATFORM) for cross-host compatibility
# - Run as current user to avoid permission issues
# - HOME set to a writable dir since the host uid has no /etc/passwd entry
#   in the container, so $HOME would otherwise resolve to / (used by pixi's
#   uv cache)
docker run -ti --rm \
    --platform "${DOCKER_PLATFORM}" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e EARTHDATA_USERNAME="${EARTHDATA_USERNAME}" \
    -e EARTHDATA_PASSWORD="${EARTHDATA_PASSWORD}" \
    -v "$(pwd)":"${CONTAINER_WORK_DIR}" \
    -v ~/.netrc:/.netrc:ro \
    --entrypoint "/bin/bash" \
    "${DOCKER_IMAGE_NAME}" \
    -l -c "eval \"\$(pixi shell-hook --frozen --manifest-path /home/ops/dist-s1/pyproject.toml)\" && cd ${CONTAINER_WORK_DIR} && python 0_generate_golden_dataset.py"

echo "Golden dataset generation completed!"
echo "Check the current directory for outputs:"
echo "- product_0/ (initial product)"
echo "- golden_dataset/ (final golden dataset)"
echo "- out_0/ and out_1/ (intermediate processing outputs)"