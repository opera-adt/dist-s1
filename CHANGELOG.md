# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.0.8] - 2025-03-05

### Changed
- Defaults to "low" for memory strategy for CPU usage.
- ProductFileData comparison to allow for individual product file comparison (fixes [#51](https://github.com/opera-adt/dist-s1/issues/51)).
- Golden dataset - used CPU and low memory strategy to create the dataset.
- Updated equality testing for DIST-S1 product comparison lowered comparison tolerance to 0.0001 (was 1e-05).
- Forced minimum version of rasterio to 1.4.0 for merging operations.
- Pydantic model updates to ensure `product_dst_dir` is set to `dst_dir` without using `setattr`.
- Updated some example parameters for testing.
- Set minimum number of workers for despeckling and estimation of normal parameters to 8.
- Logic to deal with `n_workers_for_norm_param_estimation` when GPU is available (forcing it to be 1).
- Set `batch_size_for_despeckling` to 25.

### Added
- Exposes runconfig parameter to force use of a device (`cpu`, `cuda`, `mps`, or `best`). `best` will use the best available device.
- Exposes runconfig to control batch size for despeckling (how many arrays are loaded into CPU memory at once).
- Allows for CPU multi-CPU processing (if desired) and exposes runconfig parameter to control number of workers.
   - Validates multiprocessing to use CPU device.
   - Ensures that the number of workers is not greater than the number of vCPUs (via `mp.cpu_count()`).
- If GPU is used, ensure multiprocessing is not used.
- Added a `n_workers_for_norm_param_estimation` parameter to the runconfig to control the number of workers for normal parameter estimation.
- Better instructions for generating a sample product via a docker container.
- Swap out concurrent.futures with torch.multiprocessing for normal parameter estimation for efficient CPU processing.

### Fixed
- Ensures that the number of workers for despeckling is not greater than the number of vCPUs (via `mp.cpu_count()`).
- Updates default number of parameters for CLI to match runconfig (this is what cloud operations utilize if not specified).
- removed extraneous try/except in `normal_param_estimation_workflow` used for debugging.
- Returned allclose absolute tolerance to 1e-05 for golden dataset comparison.
- Ensures Earthdata credentials are provided when localizing data and can be passed as environment variables.


## [0.0.7] - 2025-02-25

### Added
- Water mask ability to read from large water mask.
- Github issue templates (thanks to Scott Staniewicz)
- Tests for the CLI and main python interace tests.
- Minimum for distmetrics to ensure proper target crs is utilized when merging.
- Updated entrypoint for the docker container to allow for CLI arguments to be passed directly to the container.

### Fixed
- Ensures CLI correctly populates `apply_water_mask` and `water_mask_path` arguments.
- Updated the permissions of the `entrypoint.sh` file to be executable.


## [0.0.6] - 2025-02-21

### Fixed
- Issues with test_main.py related to where tmp directory was being created (solution, ensure tmp is made explicitly relative to the test directory as in `test_workflows.py`).
- All dependencies within virtual environment are back to conda-forge from PyPI.
- Product directory parameter is now correctly writing to the specified directory (fixes [#37](https://github.com/opera-adt/dist-s1/issues/37)).
- Fixed the CLI test (bug). The runconfig instance will have different product paths than the one created via the CLI because the product paths have the *processing time* in them, and that is different depending on when the runconfig object is created in the test and within the CLI-run test.

### Added
- Added a `n_workers_for_despeckling` argument to the `RunConfigData` model, CLI, and relevant processing functions.
- A test to ensure that the product directory is being correctly created and used within runconfig (added to test_main.py).


## [0.0.5] - 2025-02-19

### Fixed
- CLI issues with bucket/prefix for S3 upload (resolves [#32](https://github.com/opera-adt/dist-s1/issues/32)).
- Included `__main__.py` testing for the SAS entrypoint of the CLI; uses the cropped dataset to test the workflow.
- Includes `dist-s1 run_sas` testing and golden dataset comparision.
- Updates to README regarding GPU environment setup.

## [0.0.4]

### Added 
- Minimum working example of generation fo the product via `dist-s1 run`
- Integration of `dist-s1-enumerator` to identify/localize the inputs from MGRS tile ID, post-date, and track number
- Notebooks and examples to run end-to-end workflows as well as Science Application Software (SAS) workflows
- Docker image with nvidia compatibility (fixes the cuda version at 11.8)
- Download and application of the water mask (can specify a path or request it to generate from UMD GLAD LCLUC data).
- Extensive instructions in the README for various use-case scenarios.
- Golden dataset test for SAS workflow
- Allow user to specify bucket/prefix for S3 upload - makes library compatible with Hyp3.
- Ensure Earthdata credentials are provided in ~/.netrc and allow for them to be passed as suitable evnironment variables.
- Create a GPU compatible docker image (ongoing) - use nvidia docker image.
- Ensures pyyaml is in the environment (used for serialization of runconfig).
- Update equality testing for DIST-S1 product comparison.

### Fixed
* CLI issues with hyp3 

### Changed
- Pyproject.toml file to handle ruff

## [0.0.3]

### Added

- Python 3.13 support
- Updated dockerimage to ensure on login the conda environment is activated
- Instructions in the README for OPERA delivery.
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
