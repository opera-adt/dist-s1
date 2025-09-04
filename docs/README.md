# DIST-S1 Documentation

This directory contains the documentation for the DIST-S1 project.

## Dynamic Documentation System

The documentation uses **MkDocs with macros** to automatically generate tables from the source code. All configuration tables, product specifications, and constants are dynamically pulled from:

- `src/dist_s1/data_models/defaults.py` - Default configuration values
- `src/dist_s1/constants.py` - Product constants and specifications
- Pydantic model definitions - Field descriptions and types

### Documentation Includes:

- **RunConfigData** - Configuration for running DIST-S1 processing
- **AlgoConfigData** - Algorithm-specific configuration parameters  
- **ProductNameData** - Product naming and validation
- **Product Layers** - TIF layer specifications and NoData values
- **Disturbance Labels** - Classification labels and values

### Building the Documentation

```bash
# Install dependencies (if not already installed)
pip install mkdocs-macros-plugin

# Serve locally
mkdocs serve
```

This starts a local server (usually at http://127.0.0.1:8000) where you can view the documentation.

### File Structure

```
docs/
├── macros.py              # MkDocs macro functions for dynamic table generation
├── pages/                 # Documentation pages using macros
│   ├── config/           # Configuration documentation  
│   └── product_documentation/  # Product specification docs
└── README.md             # This file
```


