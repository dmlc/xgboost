#!/bin/bash
wget https://s3-us-west-2.amazonaws.com/xgboost-examples/MQ2008.rar
unrar x MQ2008.rar
mv -f MQ2008/Fold1/*.txt .
