FROM ghcr.io/prefix-dev/pixi:latest

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
    useradd -l -u "${UID}" -g "${GID}" --system -d /home/ops -m  -s /bin/bash dist_user

# Switch to non-root user
USER dist_user
WORKDIR /home/ops

# Ensures we cache the env install per
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#leverage-build-cache
COPY --chown=dist_user:dist_user . /home/ops/dist-s1/

# Ensure all files are read/execute by the user
RUN chmod -R a+rx /home/ops

# Create the environment from the committed lock file (also installs dist-s1 editable)
RUN pixi install --locked --manifest-path /home/ops/dist-s1/pyproject.toml && \
    pixi clean cache --yes

# Activate the pixi environment on login and interactive shells so the
# entrypoint's login shell resolves the environment's python
RUN echo 'eval "$(pixi shell-hook --frozen --manifest-path /home/ops/dist-s1/pyproject.toml)"' >> ~/.profile && \
    echo 'eval "$(pixi shell-hook --frozen --manifest-path /home/ops/dist-s1/pyproject.toml)"' >> ~/.bashrc

ENTRYPOINT ["/home/ops/dist-s1/src/dist_s1/etc/entrypoint.sh"]
CMD ["--help"]
