#!/bin/bash

rm -f *.model*

echo -e "\n ====== 1. Basic distributed-gpu test with Python: 4 workers; 1 GPU per worker ====== \n"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=4 \
  python test_gpu_basic_1x4.py

echo -e "\n ====== 2. Basic distributed-gpu test with Python: 2 workers; 2 GPUs per worker ====== \n"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=2 \
  python test_gpu_basic_2x2.py

echo -e "\n ====== 3. Basic distributed-gpu test with Python: 2 workers; Rank 0: 1 GPU, Rank 1: 3 GPUs ====== \n"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=2 \
  python test_gpu_basic_asym.py

echo -e "\n ====== 4. Basic distributed-gpu test with Python: 1 worker; 4 GPUs per worker ====== \n"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=1 \
  python test_gpu_basic_4x1.py

echo -e "\n ====== 5. RF distributed-gpu test with Python: 4 workers; 1 GPU per worker ====== \n"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=4 \
  python test_gpu_rf_1x4.py

echo -e "\n ====== 6. RF distributed-gpu test with Python: 2 workers; 2 GPUs per worker ====== \n"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=2 \
  python test_gpu_rf_2x2.py

echo -e "\n ====== 7. RF distributed-gpu test with Python: 2 workers; Rank 0: 1 GPU, Rank 1: 3 GPUs ====== \n"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=2 \
  python test_gpu_rf_asym.py

echo -e "\n ====== 8. RF distributed-gpu test with Python: 1 worker; 4 GPUs per worker ====== \n"
PYTHONPATH=../../python-package/ python ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=1 \
  python test_gpu_rf_4x1.py
