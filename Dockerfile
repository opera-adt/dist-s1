FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL description="DIST-S1 Container"

ENV DEBIAN_FRONTEND=noninteractive 
ENV CONDA_DIR=/opt/conda 
ENV LANG=C.UTF-8 
ENV LC_ALL=C.UTF-8 
ENV PATH=${CONDA_DIR}/bin:${PATH} 
ENV PYTHONDONTWRITEBYTECODE=true 
ENV PROC_HOME=/srg 
ENV MYHOME=/home/conda

ARG CONDA_UID=1000
ARG CONDA_GID=1000
ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.3.0-0

# Combine system dependency installation and cleanup into a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    vim \
    unzip \
    curl \
    gfortran && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Conda setup in a single layer with cleanup
RUN wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

# User setup
RUN groupadd -g "${CONDA_GID}" --system conda && \
    useradd -l -u "${CONDA_UID}" -g "${CONDA_GID}" --system -d /home/conda -m -s /bin/bash conda && \
    chown -R conda:conda /opt && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/conda/.profile && \
    echo "conda activate base" >> /home/conda/.profile

SHELL ["/bin/bash", "-l", "-c"]

# Switch to non-root user
USER ${CONDA_UID}
WORKDIR /home/conda

# Copy only what's needed for environment creation first
COPY --chown=${CONDA_UID}:${CONDA_GID} environment_gpu.yml /home/conda/dist-s1/

# Create conda environment and clean up in the same layer
RUN mamba env create -f /home/conda/dist-s1/environment_gpu.yml && \
    conda clean -afy

# Copy the rest of the application code
COPY --chown=${CONDA_UID}:${CONDA_GID} . /home/conda/dist-s1

# Install repository and setup environment activation
RUN echo "conda activate dist-s1-env" >> ~/.profile && \
    conda activate dist-s1-env && \
    python -m pip install --no-cache-dir /home/conda/dist-s1