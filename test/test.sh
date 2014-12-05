#!/bin/bash
if [ "$#" -lt 4 ];
then
    echo "Usage <nslave> <ndata> <config> <round_files_dir>"
    exit -1
fi

../submit_job.py $1 test_recover "${@:2}"
