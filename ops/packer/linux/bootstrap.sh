#!/bin/bash
set -euo pipefail

## Install Docker
# Add Docker's official GPG key:
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
# Allow users to use Docker without sudo
sudo usermod -aG docker ubuntu

# Start Docker daemon
sudo systemctl is-active --quiet docker.service || sudo systemctl start docker.service
sudo systemctl is-enabled --quiet docker.service || sudo systemctl enable docker.service
sleep 10  # Docker daemon takes time to come up after installing
sudo docker info

## Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sleep 10
sudo docker run --rm --gpus all ubuntu nvidia-smi
sudo systemctl stop docker

## Install AWS CLI v2
wget -nv https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -O awscliv2.zip
unzip -q awscliv2.zip
sudo ./aws/install
rm -rf ./aws/ ./awscliv2.zip

## Install jq and yq
sudo apt update && sudo apt install jq
mkdir yq/
pushd yq/
wget -nv https://github.com/mikefarah/yq/releases/download/v4.44.3/yq_linux_amd64.tar.gz -O - | \
  tar xz && sudo mv ./yq_linux_amd64 /usr/bin/yq
popd
rm -rf yq/
