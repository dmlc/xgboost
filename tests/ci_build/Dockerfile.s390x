FROM s390x/ubuntu:20.04

# Environment
ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-c"]   # Use Bash as shell

# Install all basic requirements
RUN \
    apt-get update && \
    apt-get install -y --no-install-recommends tar unzip wget git build-essential ninja-build \
      cmake time python3 python3-pip python3-numpy python3-scipy python3-sklearn r-base && \
    python3 -m pip install pytest hypothesis

ENV GOSU_VERSION 1.10

# Install lightweight sudo (not bound to TTY)
RUN set -ex; \
    wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-amd64" && \
    chmod +x /usr/local/bin/gosu && \
    gosu nobody true

# Default entry-point to use if running locally
# It will preserve attributes of created files
COPY entrypoint.sh /scripts/

WORKDIR /workspace
ENTRYPOINT ["/scripts/entrypoint.sh"]
