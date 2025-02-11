FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL description="DIST-S1 Container"

ENV DEBIAN_FRONTEND=noninteractive

ARG CONDA_UID=1000
ARG CONDA_GID=1000
ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.3.0-0

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV PYTHONDONTWRITEBYTECODE=true
ENV PROC_HOME=/srg
ENV MYHOME=/home/conda

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Conda setup
RUN apt-get update && apt-get install --no-install-recommends --yes wget bzip2 ca-certificates git > /dev/null && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

RUN apt-get install -y --no-install-recommends unzip vim curl gfortran && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN groupadd -g "${CONDA_GID}" --system conda && \
    useradd -l -u "${CONDA_UID}" -g "${CONDA_GID}" --system -d /home/conda -m  -s /bin/bash conda && \
    chown -R conda:conda /opt && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/conda/.profile && \
    echo "conda activate base" >> /home/conda/.profile

SHELL ["/bin/bash", "-l", "-c"]

# Switch to non-root user
USER ${CONDA_UID}
WORKDIR /home/conda

# Ensures we cached mamba install per
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#leverage-build-cache
COPY --chown=${CONDA_UID}:${CONDA_GID} environment_gpu.yml /home/conda/dist-s1/environment_gpu.yml
COPY --chown=${CONDA_UID}:${CONDA_GID} . /home/conda/dist-s1

# # Ensure all files are read/write by the user
# # RUN chmod -R 777 /home/ops

# Create the environment with mamba
# RUN mamba env create -f /home/conda/dist-s1/environment_gpu.yml && \
#     conda clean -afy

# Ensure that environment is activated on startup and interactive shell
RUN echo "conda activate dist-s1-env" >> ~/.profile

# Install repository with pip
RUN python -m pip install --no-cache-dir /home/conda/dist-s1