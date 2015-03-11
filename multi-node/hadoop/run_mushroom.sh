#!/bin/bash
if [ "$#" -lt 3 ];
then
	echo "Usage: <nworkers> <nthreads> <path_in_HDFS>"
	exit -1
fi

# put the local training file to HDFS
hadoop fs -mkdir $3/data
hadoop fs -put ../../demo/data/agaricus.txt.train $3/data
hadoop fs -put ../../demo/data/agaricus.txt.test $3/data

# running rabit, pass address in hdfs
../../subtree/rabit/tracker/rabit_yarn.py  -n $1 --vcores $2 ../../xgboost mushroom.hadoop.conf nthread=$2\
    data=hdfs://$3/data/agaricus.txt.train\
    eval[test]=hdfs://$3/data/agaricus.txt.test\
    model_out=hdfs://$3/mushroom.final.model

# get the final model file
hadoop fs -get $3/mushroom.final.model final.model

# output prediction task=pred 
../../xgboost mushroom.hadoop.conf task=pred model_in=final.model test:data=../../demo/data/agaricus.txt.test
# print the boosters of final.model in dump.raw.txt
../../xgboost mushroom.hadoop.conf task=dump model_in=final.model name_dump=dump.raw.txt
# use the feature map in printing for better visualization
../../xgboost mushroom.hadoop.conf task=dump model_in=final.model fmap=../../demo/data/featmap.txt name_dump=dump.nice.txt
cat dump.nice.txt
