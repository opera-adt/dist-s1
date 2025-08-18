#!/bin/bash
set -e

echo "Starting Docker-based regression test for DIST-S1 SAS"

cd "$(dirname "$0")"

# Configuration
DOCKER_IMAGE_NAME="ghcr.io/opera-adt/dist-s1:latest"
CONTAINER_WORK_DIR="/home/ops/dist-s1-data"

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

echo "Step 1: Updating runconfig for test dataset generation..."
docker run -ti --rm \
    --platform linux/amd64 \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)":"${CONTAINER_WORK_DIR}" \
    --entrypoint "/bin/bash" \
    "${DOCKER_IMAGE_NAME}" \
    -l -c "source /opt/conda/etc/profile.d/conda.sh && conda activate dist-s1-env && cd ${CONTAINER_WORK_DIR} && python 1_update_config.py"

echo "Step 2: Running DIST-S1 SAS to generate test dataset..."
docker run -ti --rm \
    --platform linux/amd64 \
    --user "$(id -u):$(id -g)" \
    -e EARTHDATA_USERNAME="${EARTHDATA_USERNAME}" \
    -e EARTHDATA_PASSWORD="${EARTHDATA_PASSWORD}" \
    -v "$(pwd)":"${CONTAINER_WORK_DIR}" \
    -v ~/.netrc:/.netrc:ro \
    --entrypoint "/bin/bash" \
    "${DOCKER_IMAGE_NAME}" \
    -l -c "source /opt/conda/etc/profile.d/conda.sh && conda activate dist-s1-env && cd ${CONTAINER_WORK_DIR} && dist-s1 run_sas --run_config_path runconfig.yml"

echo "Step 3: Finding latest OPERA_ID in golden dataset..."
if [ ! -d "golden_dataset" ]; then
    echo "Error: golden_dataset directory not found"
    exit 1
fi

GOLDEN_OPERA_ID=$(find golden_dataset -maxdepth 1 -name "OPERA_L3_DIST-ALERT-S1_*" -type d | sort | tail -1 | xargs basename)
if [ -z "$GOLDEN_OPERA_ID" ]; then
    echo "Error: No OPERA_L3_DIST-ALERT-S1_* directory found in golden_dataset"
    exit 1
fi
echo "Found golden dataset OPERA_ID: $GOLDEN_OPERA_ID"

echo "Step 4: Finding latest OPERA_ID in test dataset..."
if [ ! -d "test_product" ]; then
    echo "Error: test_product directory not found"
    exit 1
fi

TEST_OPERA_ID=$(find test_product -maxdepth 1 -name "OPERA_L3_DIST-ALERT-S1_*" -type d | sort | tail -1 | xargs basename)
if [ -z "$TEST_OPERA_ID" ]; then
    echo "Error: No OPERA_L3_DIST-ALERT-S1_* directory found in test_product"
    exit 1
fi
echo "Found test dataset OPERA_ID: $TEST_OPERA_ID"

echo "Step 5: Comparing datasets..."
GOLDEN_PATH="golden_dataset/$GOLDEN_OPERA_ID"
TEST_PATH="test_product/$TEST_OPERA_ID"

echo "Golden dataset path: $GOLDEN_PATH"
echo "Test dataset path: $TEST_PATH"

# Verify paths exist
if [ ! -d "$GOLDEN_PATH" ]; then
    echo "Error: Golden dataset path does not exist: $GOLDEN_PATH"
    exit 1
fi

if [ ! -d "$TEST_PATH" ]; then
    echo "Error: Test dataset path does not exist: $TEST_PATH"
    exit 1
fi

echo "Running comparison: dist_s1 check_equality $GOLDEN_PATH $TEST_PATH"

# Run the equality check in Docker container and capture the exit code
if docker run -ti --rm \
    --platform linux/amd64 \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)":"${CONTAINER_WORK_DIR}" \
    --entrypoint "/bin/bash" \
    "${DOCKER_IMAGE_NAME}" \
    -l -c "source /opt/conda/etc/profile.d/conda.sh && conda activate dist-s1-env && cd ${CONTAINER_WORK_DIR} && dist_s1 check_equality \"$GOLDEN_PATH\" \"$TEST_PATH\""; then
    echo "✓ SUCCESS: Datasets are equal!"
    echo "Regression test PASSED - datasets match perfectly"
else
    echo "✗ FAILURE: Datasets are NOT equal!"
    echo "Regression test FAILED - datasets do not match"
    exit 1
fi

echo "=========================================="
echo "Regression test completed successfully!"
echo "Golden: $GOLDEN_OPERA_ID"
echo "Test:   $TEST_OPERA_ID"
echo "Result: DATASETS ARE EQUAL"
echo "=========================================="