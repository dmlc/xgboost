#!/bin/bash
if [[ $# -lt 1 ]]
then
    echo "Usage: nprocess"
    exit -1
fi

rm -rf *.model
k=$1

../../tracker/rabit_demo.py -n $k linear.mock ../data/agaricus.txt.train  "${*:2}" reg_L1=1 mock=0,1,1,0 mock=1,1,1,0  mock=0,2,1,1
