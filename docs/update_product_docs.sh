#!/bin/bash

# Script to regenerate DIST-S1 product documentation tables

echo "Regenerating DIST-S1 product documentation tables..."

# Activate conda environment if available
if command -v conda &> /dev/null; then
    conda activate dist-s1-env
fi

# Generate the tables
python docs/generate_product_tables.py

echo "Product documentation tables updated successfully!"
echo "Files generated:"
echo "  - docs/product_documentation.md (combined view)"
echo "  - docs/product_layers.md (layers only)"
echo "  - docs/disturbance_labels.md (labels only)"
