#!/bin/bash

# This script is only made for running XGBoost tests on official CI where we have access
# to a 4-GPU cluster, the discovery command is for running tests on a local machine where
# the driver and the GPU worker might be the same machine for the ease of development.

if ! command -v nvidia-smi &> /dev/null
then
    # default to 4 GPUs
    echo "{\"name\":\"gpu\",\"addresses\":[\"0\",\"1\",\"2\",\"3\"]}"
    exit
else
    # https://github.com/apache/spark/blob/master/examples/src/main/scripts/getGpusResources.sh
    ADDRS=`nvidia-smi --query-gpu=index --format=csv,noheader | sed -e ':a' -e 'N' -e'$!ba' -e 's/\n/","/g'`
    echo {\"name\": \"gpu\", \"addresses\":[\"$ADDRS\"]}
fi
