#!/bin/bash

# train
../../xgboost ./conf/train.conf

# predict
../../xgboost ./conf/predict.conf

# dump
../../xgboost ./conf/dump.conf

