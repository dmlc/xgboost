ARG CUDA_VERSION
FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu18.04

# Environment
ENV DEBIAN_FRONTEND noninteractive

# Install all basic requirements
RUN \
    apt-get update && \
    apt-get install -y tar unzip wget git build-essential python3 python3-pip llvm-7 clang-tidy-7 clang-7 && \
    wget -nv -nc https://cmake.org/files/v3.12/cmake-3.12.0-Linux-x86_64.sh --no-check-certificate && \
    bash cmake-3.12.0-Linux-x86_64.sh --skip-license --prefix=/usr

# Set default clang-tidy version
RUN \
    update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-7 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-7 100

# Install Python packages
RUN \
    pip3 install pyyaml

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
