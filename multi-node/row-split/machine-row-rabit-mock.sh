#!/bin/bash
if [[ $# -ne 1 ]]
then
    echo "Usage: nprocess"
    exit -1
fi

rm -rf train-machine.row* *.model
k=$1
# make machine data
cd ../../demo/regression/
python mapfeat.py
python mknfold.py machine.txt 1
cd -

# split the lib svm file into k subfiles
python splitrows.py ../../demo/regression/machine.txt.train train-machine $k

# run xgboost mpi
../../subtree/rabit/tracker/rabit_demo.py -n $k  ../../xgboost.mock machine-row.conf dsplit=row num_round=3 mock=1,1,1,0  mock=0,0,3,0 mock=2,2,3,0
