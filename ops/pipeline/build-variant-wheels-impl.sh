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

python -m pip install -v \
  git+https://github.com/wheelnext/pep_xxx_wheel_variants.git@f3b287090f8a6f510b0e1723896e1c7e638f6bff#subdirectory=pep_xxx_wheel_variants
pip config set --site global.index-url https://variants-index.wheelnext.dev/
variantlib make-variant -f python-package/dist/xgboost-*.whl \
  -p "nvidia :: cuda :: 12" -o . --pyproject-toml python-package/pyproject.toml
