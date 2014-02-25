#!/bin/bash
python mapfeat.py
python mknfold.py agaricus.txt 1
../../xgboost mushroom.conf
