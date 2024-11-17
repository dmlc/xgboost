#!/bin/bash
set -euo pipefail

## Install basic tools
echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections
sudo apt-get update
sudo apt-get install -y cmake git build-essential wget ca-certificates curl unzip

## Install CUDA Toolkit 12.6 (Driver will be installed later)
wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6 cuda-drivers-565
rm cuda-keyring_1.1-1_all.deb
