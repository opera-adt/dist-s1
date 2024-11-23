# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3]

### Added

- Python 3.13 support
- Updated dockerimage to ensure on login the conda environment is activated
- Instructions in the README for testing outputs using the associated docker image
- A `.Dockerignore` file to remove extraneous files from the docker image
- Allow `/home/ops` directory in Docker image to be open to all users

## [0.0.2]

### Added

- Pypi delivery workflow
- Entrypoint for CLI to localize data via internet (the SAS workflow is assumed not to have internet access)
- Data models for output data and product naming conventions
- Ensures output products follow the product and the tif layers follow the expected naming conventions
  - Provides testing/validation of the structure (via tmp directories)

### Changed

- CLI entrypoints now utilize `dist-s1 run_sas` and `dist-s1 run` rathern than just `dist-s1`. 
  - The `dist-s1 run_sas` is the primary entrypoint for Science Application Software (SAS) for SDS operations. 
  - The `dist-s1 run` is the simplified entrypoint for external users, allowing for the localization of data from publicly available data sources.

## [0.0.1]

### Added

- Initial internal release of the DIST-S1 project. Test github release workflow
