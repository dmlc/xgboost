#!/bin/bash
set -e
set -x

# Download cached Maven repository, to speed up build
python3 -m awscli s3 cp s3://xgboost-ci-jenkins-artifacts/maven-repo-cache.tar.bz2 /cache/maven-repo-cache.tar.bz2 || true

if [[ -f "/cache/maven-repo-cache.tar.bz2" ]]
then
  tar xvf /cache/maven-repo-cache.tar.bz2 -C ${HOME}
fi
