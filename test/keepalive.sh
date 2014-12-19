#!/bin/bash
if [ "$#" -lt 1 ];
then
    echo "Usage: program parameters"
    echo "Repeatively run program until success"
    exit -1
fi
nrep=0
echo ./$@ rabit_task_id=$OMPI_COMM_WORLD_RANK
until ./$@ rabit_task_id=$OMPI_COMM_WORLD_RANK repeat=$nrep; do
    sleep 1
    nrep=$((nrep+1))
    echo ./$@ rabit_task_id=$OMPI_COMM_WORLD_RANK repeat=$nrep
done
