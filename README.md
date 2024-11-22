# dist-s1

[![PyPI license](https://img.shields.io/pypi/l/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![PyPI version](https://img.shields.io/pypi/v/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/dist_s1)](https://anaconda.org/conda-forge/dist_s1)
[![Conda platforms](https://img.shields.io/conda/pn/conda-forge/dist_s1)](https://anaconda.org/conda-forge/dist_s1)

This is the workflow that generates OPERA's DIST-S1 product. This workflow is designed to delineate *generic* disturbance from a time-series of OPERA Radiometric and Terrain Corrected Sentinel-1 (OPERA RTC-S1) products. The output DIST-S1 product is resampled to a 30 meter Military Grid Reference System (MGRS) tile.

Currently, this workflow is just *scaffolding*. It is not ready for use!

## Installation

### `pip`

We recommend using the mamba/conda package manager to install the DIST-S1 workflow, manage the environment, and install the dependencies.

```
mamba update -f environment.yml
pip install dist-s1  # update to conda when it is ready on conda-forge
conda activate dist-s1-env
python -m ipykernel install --user --name dist-s1-env
```

The last command is optional, but will allow this project to be imported into a Jupyter notebook.


### Development Installation

As above, we recommend using the mamba/conda package manager to install the DIST-S1 workflow, manage the environment, and install the dependencies.

```
mamba update -f environment.yml
pip install -e .
conda activate dist-s1-env
python -m ipykernel install --user --name dist-s1-env```

## Usage

There are two entrypoints for the DIST-S1 workflow:

1. `dist-s1 run_sas` - This is the primary entrypoint for Science Data System (SDS) operations in which this library is viewed as the Science Application Software (SAS) for DIST-S1 within JPL's Hybrid Science Data System (HySDS).
2. `dist-s1 run` - This is the simplified entrypoint for scientific and research users (non-SDS), allowing for the localization of data from publicly available data sources with more human readable inputs.

It is worth noting that the SDS workflow (`dist-s1 run_sas`) is *not* user friendly requiring the explicit specification of the numerous input RTC-S1 datasets (nominally there are 100s of these files required for the generation of a single DIST-S1 product over an MGRS tile). The `dist-s1 run` entrypoint has far fewer inputs and is designed to be human-operable. Specifically, the `dist-s1 run` takes care of the localization and accounting of the all the necessary input RTC-S1 datasets.

### `dist-s1 run_sas`

```
dist-s1 --runconfig_yml_path <path/to/runconfig.yml>
```

See `tests/test_main.py` for an example of how to use the CLI with sample data.

### `dist-s1 run`

This is not yet implemented.

## Docker

### Downloading a Docker Image

```
docker pull ghcr.io/asf-hyp3/dist-s1:<tag>
```
Where `<tag>` is the semantic version of the release you want to download.

Notes: 
- our image does not currently support the `arm64` (i.e. Mac M1) architecture. Therefore, you will need to build the image from the Dockerfile yourself.
- Currently, the image is still under development and we will likely update it to ensure compatibility with GPU processing.

### Building the Docker Image Locally

Make sure you have Docker installed for [Mac](https://docs.docker.com/desktop/setup/install/mac-install/) or [Windows](https://docs.docker.com/desktop/setup/install/windows-install/). We call the docker image `dist_s1_img` for the remainder of this README.

```
docker build -f Dockerfile -t dist_s1_img .
```

### Running the Container Interactively

To run the container interactively:
```
docker run -ti dist_s1_img
```
Within the container, you can run the CLI commands and the test suite.

### Inspecting Outputs from the Image

All the of the test data is currently stored in our test suite within this repostiroy and is run automatically with each PR/merge/release.
However, to allow for additional external/SDS testing via the published Docerk image, we share some of the relevant instructions.
We assume that a docker image (as above) has been built with the tag `dist_s1_img`.
Navigate to a new directory and run the following commands:
```
docker cp $(docker create dist_s1_img):/home/ops/dist-s1/tests ./tests
docker run -v "$PWD/tests:/home/ops/dist-s1/tests" dist_s1_img bash -l -c "cd dist-s1/tests && dist-s1 run_sas --runconfig_yml_path test_data/10SGD_cropped/runconfig.yml"
``` 
You should see a `tests/` directory matching the one in this repository in your current working directory. Furthermore, there now should be a `tests/OPERA_L3_DIST-ALERT-S1_*/` directory containing the expected output of this test. This is still under development, but provides a useful way to inspect the output.
