FROM ghcr.io/prefix-dev/pixi:latest

LABEL description="DIST-S1 Container"

LABEL org.opencontainers.image.title="DIST-S1"
LABEL org.opencontainers.image.description="Processor for Disturbance derived from Sentinel-1"
LABEL org.opencontainers.image.vendor="Jet Propulsion Laboratory"
LABEL org.opencontainers.image.authors="OPERA Project Science and Algorithm Team"
LABEL org.opencontainers.image.licenses="Apache-2"
LABEL org.opencontainers.image.url="https://github.com/opera-adt/dist-s1"
LABEL org.opencontainers.image.source="https://github.com/opera-adt/dist-s1"
LABEL org.opencontainers.image.documentation="https://opera-adt.github.io/dist-s1/stable/"

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=true

# Install build-essential for C++ compiler, git for setuptools_scm, libgl1-mesa-glx, unzip, and vim
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git libgl1 libglx-mesa0 unzip vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# run commands in a bash login shell
SHELL ["/bin/bash", "-l", "-c"]

# Create non-root user/group with default inputs
ARG UID=1001
ARG GID=1001

RUN groupadd -g "${GID}" --system dist_user && \
    useradd -l -u "${UID}" -g "${GID}" --system -d /home/ops -m  -s /bin/bash dist_user && \
    chmod o+rx /home/ops

# Switch to non-root user
USER dist_user
WORKDIR /home/ops

# Ensures we cache the env install per
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#leverage-build-cache
# --chmod grants write too: any uid can run this image (e.g. `docker run
# --user "$(id -u):$(id -g)"` to match the host user for output file
# ownership), and uv re-verifies/rebuilds the editable dist-s1 package's
# egg-info on activation even with `--frozen`.
COPY --chown=dist_user:dist_user --chmod=777 . /home/ops/dist-s1/

# uv builds the editable dist-s1 in an isolated copy of the source that omits
# the .git directory, so setuptools_scm cannot infer the version there. Resolve
# it from the .git present in the build context and pass it through explicitly.
# chmod runs in the same layer as pixi install (rather than a separate RUN
# afterwards) so overlay2 doesn't copy-up the multi-GB env a second time.
RUN export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DIST_S1="$(git -C /home/ops/dist-s1 describe --tags | sed -E 's/^v//; s/-([0-9]+)-g/.post\1+g/')" && \
    pixi install --locked --manifest-path /home/ops/dist-s1/pyproject.toml && \
    pixi clean cache --yes && \
    chmod -R a+rwX /home/ops/dist-s1

# Activate the pixi environment on login and interactive shells so the
# entrypoint's login shell resolves the environment's python
RUN echo 'eval "$(pixi shell-hook --frozen --manifest-path /home/ops/dist-s1/pyproject.toml)"' >> ~/.profile && \
    echo 'eval "$(pixi shell-hook --frozen --manifest-path /home/ops/dist-s1/pyproject.toml)"' >> ~/.bashrc

ENTRYPOINT ["/home/ops/dist-s1/src/dist_s1/etc/entrypoint.sh"]
CMD ["--help"]
