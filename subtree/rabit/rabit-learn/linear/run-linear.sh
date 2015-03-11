#!/bin/bash
if [[ $# -lt 1 ]]
then
    echo "Usage: nprocess"
    exit -1
fi

rm -rf *.model
k=$1

# run linear model, the program will automatically split the inputs
../../tracker/rabit_demo.py -n $k linear.rabit ../data/agaricus.txt.train reg_L1=1 

./linear.rabit ../data/agaricus.txt.test task=pred model_in=final.model
