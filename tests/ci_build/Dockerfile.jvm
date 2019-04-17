FROM centos:6

# Install all basic requirements
RUN \
    yum -y update && \
    yum install -y tar unzip wget xz git centos-release-scl yum-utils java-1.8.0-openjdk-devel && \
    yum-config-manager --enable centos-sclo-rh-testing && \
    yum -y update && \
    yum install -y devtoolset-4-gcc devtoolset-4-binutils devtoolset-4-gcc-c++ && \
    # Python
    wget https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh && \
    bash Miniconda3-4.5.12-Linux-x86_64.sh -b -p /opt/python && \
    # CMake
    wget -nv -nc https://cmake.org/files/v3.12/cmake-3.12.0-Linux-x86_64.sh --no-check-certificate && \
    bash cmake-3.12.0-Linux-x86_64.sh --skip-license --prefix=/usr && \
    # Maven
    wget http://apache.osuosl.org/maven/maven-3/3.6.1/binaries/apache-maven-3.6.1-bin.tar.gz && \
    tar xvf apache-maven-3.6.1-bin.tar.gz -C /opt && \
    ln -s /opt/apache-maven-3.6.1/ /opt/maven

ENV PATH=/opt/python/bin:/opt/maven/bin:$PATH
ENV CC=/opt/rh/devtoolset-4/root/usr/bin/gcc
ENV CXX=/opt/rh/devtoolset-4/root/usr/bin/c++
ENV CPP=/opt/rh/devtoolset-4/root/usr/bin/cpp

# Install Python packages
RUN \
    pip install numpy pytest scipy scikit-learn wheel kubernetes urllib3==1.22 awscli

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
