#!/bin/bash

# echo "====== 1. Basic distributed-gpu test with Python: 4 workers; 1 GPU per worker ======"
# PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=4 \
#   python test_gpu_basic_1x4.py

echo "====== 2. Basic distributed-gpu test with Python: 2 workers; 2 GPUs per worker ======"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=2 \
  python test_gpu_basic_2x2.py
 
# echo "====== 3. Basic distributed-gpu test with Python: 2 workers; Rank 0: 1 GPU, Rank 1: 3 GPUs ======"
# PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=2 \
#   python test_gpu_basic_asym.py
# 
# echo "====== 4. Basic distributed-gpu test with Python: 1 worker; 4 GPUs per worker ======"
# PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=1 \
#   python test_gpu_basic_4x1.py