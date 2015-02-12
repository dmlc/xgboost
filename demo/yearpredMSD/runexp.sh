#!/bin/bash

if [ -f YearPredictionMSD.txt ]
then
    echo "use existing data to run experiment"
else
    echo "getting data from uci, make sure you are connected to internet"
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip
    unzip YearPredictionMSD.txt.zip
fi
echo "start making data.."
# map feature using indicator encoding, also produce featmap.txt
python csv2libsvm.py YearPredictionMSD.txt yearpredMSD.libsvm
head -n 463715 yearpredMSD.libsvm > yearpredMSD.libsvm.train
tail -n 51630 yearpredMSD.libsvm > yearpredMSD.libsvm.test
echo "finish making the data"
../../xgboost yearpredMSD.conf
