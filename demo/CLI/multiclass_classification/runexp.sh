#!/bin/bash
if [ -f dermatology.data ]
then
    echo "use existing data to run multi class classification"
else
    echo "getting data from uci, make sure you are connected to internet"
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data
fi
python train.py
