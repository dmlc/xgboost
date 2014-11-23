#!/bin/bash
if [[ $# -ne 1 ]]
then
    echo "Usage: nprocess"
    exit -1
fi

#
# This script is same as mushroom-col except that we will be using xgboost python module
# 
# xgboost used built in tcp-based allreduce module, and can be run on more enviroment, so long as we know how to start job by modifying ../submit_job_tcp.py
#
rm -rf train.col* *.model
k=$1

# split the lib svm file into k subfiles
python splitsvm.py ../../demo/data/agaricus.txt.train train $k

# run xgboost mpi
../submit_job_tcp.py $k python mushroom-col.py

cat dump.nice.$k.txt
