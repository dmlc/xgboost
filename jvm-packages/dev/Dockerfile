#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
FROM centos:7

# Install all basic requirements
RUN \
    yum -y update && \
    yum install -y bzip2 make tar unzip wget xz git centos-release-scl yum-utils java-1.8.0-openjdk-devel && \
    yum-config-manager --enable centos-sclo-rh-testing && \
    yum -y update && \
    yum install -y devtoolset-7-gcc devtoolset-7-binutils devtoolset-7-gcc-c++ && \
    # Python
    wget https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh && \
    bash Miniconda3-4.5.12-Linux-x86_64.sh -b -p /opt/python && \
    # CMake
    wget -nv -nc https://cmake.org/files/v3.18/cmake-3.18.3-Linux-x86_64.sh --no-check-certificate && \
    bash cmake-3.18.3-Linux-x86_64.sh --skip-license --prefix=/usr && \
    # Maven
    wget https://archive.apache.org/dist/maven/maven-3/3.6.1/binaries/apache-maven-3.6.1-bin.tar.gz && \
    tar xvf apache-maven-3.6.1-bin.tar.gz -C /opt && \
    ln -s /opt/apache-maven-3.6.1/ /opt/maven

# Set the required environment variables
ENV PATH=/opt/python/bin:/opt/maven/bin:$PATH
ENV CC=/opt/rh/devtoolset-7/root/usr/bin/gcc
ENV CXX=/opt/rh/devtoolset-7/root/usr/bin/c++
ENV CPP=/opt/rh/devtoolset-7/root/usr/bin/cpp
ENV JAVA_HOME=/usr/lib/jvm/java

# Install Python packages
RUN \
    pip install numpy pytest scipy scikit-learn wheel kubernetes urllib3==1.22 awscli

ENV GOSU_VERSION 1.10

# Install lightweight sudo (not bound to TTY)
RUN set -ex; \
    wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-amd64" && \
    chmod +x /usr/local/bin/gosu && \
    gosu nobody true

WORKDIR /xgboost
