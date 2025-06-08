FROM condaforge/miniforge3:latest

LABEL description="DIST-S1 Container"

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=true

# Install build-essential for C++ compiler, libgl1-mesa-glx, unzip, and vim
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libgl1 libglx-mesa0 unzip vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# run commands in a bash login shell
SHELL ["/bin/bash", "-l", "-c"]

# Create non-root user/group with default inputs
ARG UID=1001
ARG GID=1001

RUN groupadd -g "${GID}" --system dist_user && \
    useradd -l -u "${UID}" -g "${GID}" --system -d /home/ops -m  -s /bin/bash dist_user && \
    chown -R dist_user:dist_user /opt

# Switch to non-root user
USER dist_user
WORKDIR /home/ops

# Ensures we cached mamba install per
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#leverage-build-cache
COPY --chown=dist_user:dist_user . /home/ops/dist-s1/

# Ensure all files are read/execute by the user
RUN chmod -R a+rx /home/ops

# Create the environment with mamba
RUN mamba env create -f /home/ops/dist-s1/environment.yml && \
    conda clean -afy

# Ensure that environment is activated on startup and interactive shell
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.profile && \
    echo "conda activate dist-s1-env" >> ~/.profile
RUN echo "conda activate dist-s1-env" >> ~/.bashrc

# Install repository with pip
RUN python -m pip install --no-cache-dir /home/ops/dist-s1

ENTRYPOINT ["/home/ops/dist-s1/src/dist_s1/etc/entrypoint.sh"]
CMD ["--help"]
