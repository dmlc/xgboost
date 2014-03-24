#!/bin/bash
# map feature using indicator encoding, also produce featmap.txt
python mapfeat.py
# split train and test
python mknfold.py machine.txt 1
# training and output the models
../../xgboost machine.conf
# output predictions of test data
../../xgboost machine.conf task=pred model_in=0003.model
# print the boosters of 00003.model in dump.raw.txt
../../xgboost machine.conf task=dump model_in=0003.model name_dump=dump.raw.txt 