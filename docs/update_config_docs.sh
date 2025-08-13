#!/bin/bash
# Script to regenerate API documentation tables

echo "Regenerating API documentation tables..."

# Activate conda environment and run the generation script
conda activate dist-s1-env && python docs/generate_api_tables.py

echo "API documentation tables updated successfully!"
echo "You can now run 'mkdocs serve' to view the updated documentation."
