#!/bin/bash

mkdir -p result
# train
../../xgboost ./conf/train.conf

# predict
../../xgboost ./conf/predict.conf

# dump
../../xgboost ./conf/dump.conf

