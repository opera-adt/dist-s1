FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL description="DIST-S1 Container"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

RUN conda install -n base -c conda-forge mamba \
    && conda clean -afy

# Default command
CMD ["bash"]

# Create non-root user/group with default inputs
ARG UID=1000
ARG GID=1000

RUN groupadd -g "${GID}" --system dist_user && \
    useradd -l -u "${UID}" -g "${GID}" --system -d /home/ops -m  -s /bin/bash dist_user && \
    chown -R dist_user:dist_user /opt

# Switch to non-root user
USER dist_user
WORKDIR /home/ops

# Ensures we cached mamba install per
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#leverage-build-cache
COPY --chown=dist_user:dist_user environment_gpu.yml /home/ops/dist-s1/environment_gpu.yml
COPY --chown=dist_user:dist_user . /home/ops/dist-s1

# Ensure all files are read/write by the user
# RUN chmod -R 777 /home/ops

# Create the environment with mamba
RUN mamba env create -f /home/ops/dist-s1/environment_gpu.yml && \
    conda clean -afy

# Ensure that environment is activated on startup and interactive shell
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.profile && \
    echo "conda activate dist-s1-env" >> ~/.profile
RUN echo "conda activate dist-s1-env" >> ~/.bashrc

# Install repository with pip
RUN python -m pip install --no-cache-dir /home/ops/dist-s1
