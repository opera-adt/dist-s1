# DIST-S1 Documentation

This directory contains the documentation for the DIST-S1 project.

## API Documentation

The API documentation is dynamically generated from the Pydantic models in the codebase and includes:

- **RunConfigData** - Configuration for running DIST-S1 processing
- **AlgoConfigData** - Algorithm-specific configuration parameters

### Regenerating Documentation

```bash
# Use the convenience script
./docs/update_api_docs.sh

# Or run manually
conda activate dist-s1-env
python docs/generate_api_tables.py
```

### Building the Documentation

```bash
conda activate dist-s1-env
mkdocs serve
```

This starts a local server (usually at http://127.0.0.1:8000) where you can view the documentation.

### CI/CD Integration

The documentation generation is integrated into the CI/CD pipeline:
- **Automatic Testing**: Ensures documentation generation works correctly
- **Automatic Deployment**: Generates fresh documentation before deploying to GitHub Pages
- **Manual Updates**: Can be triggered via GitHub Actions to update documentation files

For manual updates, go to GitHub Actions → "Update API Documentation" workflow → "Run workflow".


