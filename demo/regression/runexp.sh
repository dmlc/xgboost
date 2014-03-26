#!/bin/bash
# map the data to features. For convenience we only use 7 original attributes and encode them as features in a trivial way 
python mapfeat.py
# split train and test
python mknfold.py machine.txt 1
# training and output the models
../../xgboost machine.conf
# output predictions of test data
../../xgboost machine.conf task=pred model_in=0002.model
# print the boosters of 00002.model in dump.raw.txt
../../xgboost machine.conf task=dump model_in=0002.model name_dump=dump.raw.txt 