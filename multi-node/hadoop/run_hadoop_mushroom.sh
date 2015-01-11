#!/bin/bash

if [ "$#" -lt 2 ];
then
    echo "Usage: <num_of_slave_nodes> <path_in_HDFS>"
    exit -1
fi

# put the local training file to HDFS
hadoop fs -mkdir $2/data
hadoop fs -put ../../demo/data/agaricus.txt.train $2/data

# training and output the final model file
../../rabit/tracker/rabit_hadoop.py -n $1 -i $2/data/agaricus.txt.train \
    -o $2/model -f ../../demo/data/agaricus.txt.test \
    ../../xgboost mushroom.hadoop.conf dsplit=row 

# get the final model file
hadoop fs -get $2/model/part-00000 ./final.model

# output prediction task=pred 
../../xgboost mushroom.hadoop.conf task=pred model_in=final.model \
    test:data=../../demo/data/agaricus.txt.test
# print the boosters of final.model in dump.raw.txt
../../xgboost mushroom.hadoop.conf task=dump model_in=final.model name_dump=dump.raw.txt
# use the feature map in printing for better visualization
../../xgboost mushroom.hadoop.conf task=dump model_in=final.model \
    fmap=../../demo/data/featmap.txt name_dump=dump.nice.txt
cat dump.nice.txt
