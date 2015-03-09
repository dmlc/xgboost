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

# submit to hadoop
../../tracker/rabit_yarn.py  -n $1 --vcores 1 linear.rabit hdfs://$2/data/agaricus.txt.train model_out=hdfs://$2/mushroom.linear.model "${*:3}" 

# get the final model file
hadoop fs -get $2/mushroom.linear.model ./linear.model

./linear.rabit ../data/agaricus.txt.test task=pred model_in=linear.model
