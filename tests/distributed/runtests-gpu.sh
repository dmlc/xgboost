#!/bin/bash

rm -f *.model*

export DMLC_SUBMIT_CLUSTER=local
submit="timeout 30 python ../../dmlc-core/tracker/dmlc-submit"

echo -e "\n ====== 1. Basic distributed-gpu test with Python: 4 workers; 1 GPU per worker ====== \n"
$submit --num-workers=4 python distributed_gpu.py basic_1x4 || exit 1

echo -e "\n ====== 2. Basic distributed-gpu test with Python: 2 workers; 2 GPUs per worker ====== \n"
$submit --num-workers=2 python distributed_gpu.py basic_2x2 || exit 1

echo -e "\n ====== 3. Basic distributed-gpu test with Python: 2 workers; Rank 0: 1 GPU, Rank 1: 3 GPUs ====== \n"
$submit --num-workers=2 python distributed_gpu.py basic_asym || exit 1

echo -e "\n ====== 4. Basic distributed-gpu test with Python: 1 worker; 4 GPUs per worker ====== \n"
$submit --num-workers=1 python distributed_gpu.py basic_4x1 || exit 1

echo -e "\n ====== 5. RF distributed-gpu test with Python: 4 workers; 1 GPU per worker ====== \n"
$submit --num-workers=4 python distributed_gpu.py rf_1x4 || exit 1

echo -e "\n ====== 6. RF distributed-gpu test with Python: 2 workers; 2 GPUs per worker ====== \n"
$submit --num-workers=2 python distributed_gpu.py rf_2x2 || exit 1

echo -e "\n ====== 7. RF distributed-gpu test with Python: 2 workers; Rank 0: 1 GPU, Rank 1: 3 GPUs ====== \n"
$submit --num-workers=2 python distributed_gpu.py rf_asym || exit 1

echo -e "\n ====== 8. RF distributed-gpu test with Python: 1 worker; 4 GPUs per worker ====== \n"
$submit --num-workers=1 python distributed_gpu.py rf_4x1 || exit 1
