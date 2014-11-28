#!/bin/bash
if [ "$#" -ne 3 ];
then
    echo "Usage <nslave> <ndata> <config>"
    exit -1
fi
../submit_job_tcp.py $1 test_allreduce $2 $3
