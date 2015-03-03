#!/bin/bash
if [ "$#" -lt 3 ];
then
	echo "Usage: <nworkers> <path_in_HDFS> [param=val]"
	exit -1
fi

# put the local training file to HDFS
hadoop fs -rm -r -f $2/data
hadoop fs -rm -r -f $2/mushroom.linear.model
hadoop fs -mkdir $2/data
hadoop fs -put ../data/agaricus.txt.train $2/data

# submit to hadoop
../../tracker/rabit_hadoop.py --host_ip ip  -n $1 -i $2/data/agaricus.txt.train -o $2/mushroom.linear.model linear.rabit stdin model_out=stdout "${*:3}" 

# get the final model file
hadoop fs -get $2/mushroom.linear.model/part-00000 ./linear.model

./linear.rabit ../data/agaricus.txt.test task=pred model_in=linear.model
