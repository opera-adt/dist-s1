# dist-s1

[![PyPI license](https://img.shields.io/pypi/l/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![PyPI version](https://img.shields.io/pypi/v/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/dist_s1)](https://anaconda.org/conda-forge/dist_s1)
[![Conda platforms](https://img.shields.io/conda/pn/conda-forge/dist_s1)](https://anaconda.org/conda-forge/dist_s1)

This is the workflow that generates OPERA's DIST-S1 product. This workflow is designed to delineate *generic* disturbance from a time-series of OPERA Radiometric and Terrain Corrected Sentinel-1 (RTC-S1) products.

## Install

```
mamba update -f environment.yml
pip install -e .
conda activate dist-s1-env
python -m ipykernel install --user --name dist-s1-env
```

## Usage

```
dist-s1 --runconfig_yml_path <path/to/runconfig.yml>
```

See `tests/test_main.py` for an example of how to use the CLI with sample data.


## Docker

### Downloading a Docker Image

```
docker pull ghcr.io/asf-hyp3/dist-s1:<tag>
```
Where `<tag>` is the semantic version of the release you want to download.

### Building the Docker Image Locally

Make sure you have Docker installed (e.g. for MacOS: https://docs.docker.com/desktop/setup/install/mac-install/)

```
docker build -f Dockerfile -t dist_s1_img .
```

### Running the Container Interactively

To run the container interactively:
```
docker run -ti dist_s1_img
```
Within the container, you can run the CLI commands and the test suite.
