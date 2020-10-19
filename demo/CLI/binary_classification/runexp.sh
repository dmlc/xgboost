#!/bin/bash
# map feature using indicator encoding, also produce featmap.txt
python mapfeat.py
# split train and test
python mknfold.py agaricus.txt 1

XGBOOST=../../../xgboost

# training and output the models
$XGBOOST mushroom.conf
# output prediction task=pred
$XGBOOST mushroom.conf task=pred model_in=0002.model
# print the boosters of 00002.model in dump.raw.txt
$XGBOOST mushroom.conf task=dump model_in=0002.model name_dump=dump.raw.txt
# use the feature map in printing for better visualization
$XGBOOST mushroom.conf task=dump model_in=0002.model fmap=featmap.txt name_dump=dump.nice.txt
cat dump.nice.txt
