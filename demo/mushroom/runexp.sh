#!/bin/bash
python mapfeat.py
python mknfold.py agaricus.txt 1
../../xgboost mushroom.conf
../../xgboost mushroom.conf task=dump model_in=0003.model
../../xgboost mushroom.conf task=dumppath model_in=0003.model 
python maptree.py
