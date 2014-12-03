#!/bin/bash
if [ "$#" -lt 1 ];
then
    echo "Usage: program parameters"
    echo "Repeatively run program until success"
    exit -1
fi
echo ./$@ job_id=$OMPI_COMM_WORLD_RANK
until ./$@ job_id=$OMPI_COMM_WORLD_RANK; do
    echo "Server "$1" crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
