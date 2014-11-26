#!/bin/bash
if [ "$#" -ne 2 ];
then
    echo "Usage <nslave> <ndata>"
    exit -1
fi
../submit_job_tcp.py $1 test_allreduce $2
