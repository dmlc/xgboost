#!/bin/bash
if [ "$#" -lt 5 ];
then
    echo "Usage: <nslaves> <input_data> <ncluster> <max_iteration> <output>"
    exit -1
fi
python ../rabit_hadoop.py -s $1 -i $2  -m kmeans --args "stdin "$3" "$4" stdout" -o $5

 