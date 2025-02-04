# dist-s1

[![PyPI license](https://img.shields.io/pypi/l/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![PyPI version](https://img.shields.io/pypi/v/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/dist_s1)](https://anaconda.org/conda-forge/dist_s1)
[![Conda platforms](https://img.shields.io/conda/pn/conda-forge/dist_s1)](https://anaconda.org/conda-forge/dist_s1)

This is the workflow that generates OPERA's DIST-S1 product. This workflow is designed to delineate *generic* disturbance from a time-series of OPERA Radiometric and Terrain Corrected Sentinel-1 (OPERA RTC-S1) products. The output DIST-S1 product is resampled to a 30 meter Military Grid Reference System (MGRS) tile.

## Installation

We recommend using the mamba/conda package manager and `conda-forge` distributions to install the DIST-S1 workflow, manage the environment, and install the dependencies.

```
mamba update -f environment.yml
pip install dist-s1  # update to conda when it is ready on conda-forge
conda activate dist-s1-env
```

The last command is optional, but will allow this project to be imported into a Jupyter notebook.

### GPU Installation

We have tried to make the environment as open, flexible, and transparent as possible. 
In particular, we are using the `conda-forge` distribution of the libraries, including relevant GPU-drivers.
We have provided an `environment_gpu.yml` which simply adds a minimum version of conda-forge's `cudatoolkit` to the environment to ensure on our systems that GPU is accessible.
This will *not* be installable on non-Linux systems (and maybe even non-Linux systems without GPUs).
The library `cudatoolkit` is the `conda-forge` distribution of NVIDIA's cuda tool kit (see [here](https://anaconda.org/conda-forge/cudatoolkit)).
That said, there could be instances when a new library will introduce a CPU-only version and will break the ability to use GPU on Linux.

To resolve environment issues related to having access to the GPU, we successfully used `conda-tree` to identify CPU bound dependencies.
For example,
```
mamba install -c conda-forge conda-tree
conda-tree -n dist-s1-env deptree
```
We then identified packages with `mkl` and `cpu` in their distribution names.
There may be other libraries or methods of using `conda-tree` that are more elegant and efficient.
That said, the above provides an avenue for identifying such issues with the environment.


### Jupyter Kernel

For the Jupyter notebooks, install the jupyter dependencies:
```
mamba install jupyterlab ipywidgets black isort jupyterlab_code_formatter 
```
We also install the kernel `dist-s1-env` using the environment above via:
```
python -m ipykernel install --user --name dist-s1-env
```

### Development Installation

As above, we recommend using the mamba/conda package manager to install the DIST-S1 workflow, manage the environment, and install the dependencies.

```
mamba update -f environment.yml
pip install -e .
conda activate dist-s1-env
python -m ipykernel install --user --name dist-s1-env
```

## Usage

There are two entrypoints for the DIST-S1 workflow:

1. `dist-s1 run_sas` - This is the primary entrypoint for Science Data System (SDS) operations in which this library is viewed as the Science Application Software (SAS) for DIST-S1 within JPL's Hybrid Science Data System (HySDS).
2. `dist-s1 run` (not implemented yet) - This is the simplified entrypoint for scientific and research users (non-SDS), allowing for the localization of data from publicly available data sources with more human readable inputs.

It is worth noting that the SDS workflow (`dist-s1 run_sas`) is *not* user friendly requiring the explicit specification of the numerous input RTC-S1 datasets (nominally there are 100s of these files required for the generation of a single DIST-S1 product over an MGRS tile). The `dist-s1 run` entrypoint has far fewer inputs and is designed to be human-operable. Specifically, the `dist-s1 run` takes care of the localization and accounting of the all the necessary input RTC-S1 datasets.

### The `dist-s1 run_sas` Entrypoint

```
dist-s1 run_sas --runconfig_yml_path <path/to/runconfig.yml>
```

See `tests/test_main.py` for an example of how to use the CLI with sample data.

### The `dist-s1 run` Entrypoint

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
docker build -f Dockerfile -t dist-s1 .
```

### Running the Container Interactively

To run the container interactively:
```
docker run -ti dist-s1
```
Within the container, you can run the CLI commands and the test suite.


# Delivery Instructions

There are certain releases associated with OPERA project deliveries. Here we provide instructions for how to run and verify the DIST-S1 workflow.

We have included sample input data, associated a Docker image via the Github registry, and run tests via github actions all within this repository.

```
docker pull ghcr.io/opera-adt/dist-s1
```
If a specific version is required (or assumed for a delivery), then you use `docker pull ghcr.io/opera-adt/dist-s1:<version>` e.g.
```
docker pull ghcr.io/opera-adt/dist-s1:0.0.4
```
The command will pull the latest released version of the Docker image. To run the test suite, run:
```
docker run ghcr.io/opera-adt/dist-s1 bash -l -c 'cd dist-s1 && pytest tests'
``` 


