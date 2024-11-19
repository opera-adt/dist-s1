# dist-s1

[![PyPI license](https://img.shields.io/pypi/l/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![PyPI version](https://img.shields.io/pypi/v/dist-s1.svg)](https://pypi.python.org/pypi/dist-s1/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/dist_s1)](https://anaconda.org/conda-forge/dist_s1)
[![Conda platforms](https://img.shields.io/conda/pn/conda-forge/dist_s1)](https://anaconda.org/conda-forge/dist_s1)

This is the workflow that generates OPERA's DIST-S1 product.

## Install

```
mamba update -f environment.yml
pip install -e .
conda activate dist-s1-env
python -m ipykernel install --user --name dist-s1-env
```

## Docker

### Local Usage

Make sure you have Docker installed (e.g. for MacOS: https://docs.docker.com/desktop/setup/install/mac-install/)

```
docker build -f Dockerfile -t dist_s1_img .
```
To run the container interactively:
```
docker run -ti dist_s1_img
```
Should be able to run the CLI commands and the test suite.
