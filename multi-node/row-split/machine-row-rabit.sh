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
../../subtree/rabit/tracker/rabit_demo.py -n $k ../../xgboost machine-row.conf dsplit=row num_round=3 eval_train=1

# run xgboost-mpi save model 0001, continue to run from existing model
../../subtree/rabit/tracker/rabit_demo.py -n $k ../../xgboost machine-row.conf dsplit=row num_round=1
../../subtree/rabit/tracker/rabit_demo.py -n $k ../../xgboost machine-row.conf dsplit=row num_round=2 model_in=0001.model
