#!/bin/bash

rm -f *.model*

export DMLC_SUBMIT_CLUSTER=local
submit="timeout 30 python ../../dmlc-core/tracker/dmlc-submit"

echo -e "\n ====== 1. Basic distributed-gpu test with Python: 4 workers; 1 GPU per worker ====== \n"
$submit --num-workers=$(nvidia-smi -L | wc -l) python distributed_gpu.py basic_1x4 || exit 1

echo -e "\n ====== 2. RF distributed-gpu test with Python: 4 workers; 1 GPU per worker ====== \n"
$submit --num-workers=$(nvidia-smi -L | wc -l) python distributed_gpu.py rf_1x4 || exit 1
