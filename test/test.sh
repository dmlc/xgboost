#!/bin/bash
if [ "$#" -ne 4 ];
then
    echo "Usage <nslave> <ndata> <config> <round_files_dir>"
    exit -1
fi

../submit_job_tcp.py $1 test_recover $2 $3 $4 $5
