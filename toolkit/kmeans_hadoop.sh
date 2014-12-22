#!/bin/bash
if [ "$#" -lt 5 ];
then
    echo "Usage: <nslaves> <input_data> <ncluster> <max_iteration> <output>"
    exit -1
fi
#set path to hadoop streaming jar here
STREAMING_JAR=
python ../tracker/rabit_hadoop.py -hs $STREAMING_JAR -s $1 -i $2  -m kmeans.rabit --args "stdin "$3" "$4" stdout" -o $5 --file kmeans.rabit
