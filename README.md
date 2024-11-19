# dist-s1

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/opera-adt/dist-s1/workflows/CI/badge.svg
[actions-link]:             https://github.com/opera-adt/dist-s1/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/dist-s1
[conda-link]:               https://github.com/conda-forge/dist-s1-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/opera-adt/dist-s1/discussions
[pypi-link]:                https://pypi.org/project/dist-s1/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/dist-s1
[pypi-version]:             https://img.shields.io/pypi/v/dist-s1

<!-- prettier-ignore-end -->

This is the workflow that generates OPERA's DIST-S1 product.

## Install

```
mamba update -f environment.yml
pip install -e .
conda activate dist-s1-env
python -m ipykernel install --user --name dist-s1-env
```