#!/bin/bash
if [ "$#" -lt 4 ];
then
    echo "Usage <nslave> <k> <d> <itr> <data_dir>"
    exit -1
fi

../submit_job.py $1 kmeans "${@:2}"
