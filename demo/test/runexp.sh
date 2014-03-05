#!/bin/bash
# map feature using indicator encoding, also produce featmap.txt
python mapfeat.py
# split train and test
python mknfold.py agaricus.txt 1
# training
../../xgboost mushroom.conf
# this is what dump will looklike without feature map
../../xgboost mushroom.conf task=dump model_in=0003.model name_dump=dump.raw.txt 
# this is what dump will looklike with feature map
../../xgboost mushroom.conf task=dump model_in=0003.model fmap=featmap.txt name_dump=dump.nice.txt
cat dump.nice.txt
