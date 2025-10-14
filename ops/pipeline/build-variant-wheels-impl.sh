#!/bin/bash
## Build Python wheels using Wheel Variant prototype (WheelNext)
## Companion script for ops/pipeline/build-variant-wheels.sh

set -eo pipefail

set -x
gosu root chown -R $(id -u):$(id -g) /opt/miniforge/envs /opt/miniforge/pkgs/cache
gosu root chown $(id -u):$(id -g) /opt/miniforge/pkgs
set +x

mamba create -y -n wheelnext python=3.13 python-build

source activate wheelnext

# Cannot set -u before Conda env activation
set -xu

python -m pip install "variantlib[cli] @ git+https://github.com/wheelnext/variantlib.git@main"
python -m pip install "nvidia-variant-provider @ git+https://github.com/wheelnext/nvidia-variant-provider.git@master"
variantlib make-variant --no-isolation -f python-package/dist/xgboost-*.whl \
  -p "nvidia :: cuda_version_lower_bound :: 12.0" \
  -p "nvidia :: cuda_version_upper_bound :: 13" \
  -o . --pyproject-toml python-package/pyproject.toml
