#!/bin/bash
set -e

echo "Starting Docker-based regression test for DIST-S1 SAS (Docker Container Version)"

# Function to run commands in Docker container
run_in_docker() {
    local cmd="$1"
    echo "Running in Docker: $cmd"
    docker run --rm \
        -v "$(pwd)":/workspace \
        -w /workspace \
        dist-s1:latest \
        bash -c "$cmd"
}

cd "$(dirname "$0")"

echo "Step 1: Running DIST-S1 SAS to generate test dataset in Docker..."
run_in_docker "dist-s1 run_sas --run_config_path runconfig.yml"

echo "Step 2: Finding latest OPERA_ID in golden dataset..."
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

echo "Step 3: Finding latest OPERA_ID in test dataset..."
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

echo "Step 4: Comparing datasets in Docker..."
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

# Run the equality check and capture the exit code
if run_in_docker "dist_s1 check_equality $GOLDEN_PATH $TEST_PATH"; then
    echo "✓ SUCCESS: Datasets are equal!"
    echo "Regression test PASSED - datasets match perfectly"
else
    echo "✗ FAILURE: Datasets are NOT equal!"
    echo "Regression test FAILED - datasets do not match"
    exit 1
fi

echo "=========================================="
echo "Docker-based regression test completed successfully!"
echo "Golden: $GOLDEN_OPERA_ID"
echo "Test:   $TEST_OPERA_ID"
echo "Result: DATASETS ARE EQUAL"
echo "=========================================="