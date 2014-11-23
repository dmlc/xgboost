Distributed XGBoost
======
This folder contains information about experimental version of distributed xgboost.

Build
=====
* In the root folder, run ```make mpi```, this will give you xgboost-mpi
  - You will need to have MPI to build xgboost-mpi
* Alternatively, you can run ```make```, this will give you xgboost, which uses a beta buildin allreduce
  - You do not need MPI to build this, you can modify [submit_job_tcp.py](submit_job_tcp.py) to use any job scheduler you like to submit the job

Design Choice
=====
* Does distributed xgboost must reply on MPI library?
  - No, XGBoost replies on MPI protocol that provide Broadcast and AllReduce,
  - The dependency is isolated in [sync module](../src/sync/sync.h)
  - All other parts of code uses interface defined in sync.h
  - [sync_mpi.cpp](../src/sync/sync_mpi.cpp) is a implementation of sync interface using standard MPI library, to use xgboost-mpi, you need an MPI library
  - If there are platform/framework that implements these protocol, xgboost should naturally extends to these platform
  - As an example, [sync_tcp.cpp](../src/sync/sync_tcp.cpp) is an implementation of interface using TCP, and is linked with xgboost by default

* How is the data distributed?
  - There are two solvers in distributed xgboost
  - Column-based solver split data by column, each node work on subset of columns, 
    it uses exactly the same algorithm as single node version.
  - Row-based solver split data by row, each node work on subset of rows,
    it uses an approximate histogram count algorithm, and will only examine subset of 
    potential split points as opposed to all split points.


Usage
====
* You will need a network filesystem, or copy data to local file system before running the code
* xgboost-mpi run in MPI enviroment, 
* xgboost can be used together with [submit_job_tcp.py](submit_job_tcp.py) on other types of job schedulers
* ***Note*** The distributed version is still multi-threading optimized.
    You should run one process per node that takes most available CPU,
    this will reduce the communication overhead and improve the performance.
   - One way to do that is limit mpi slot in each machine to be 1, or reserve nthread processors for each process.
* Examples:
  - [Column-based version](col-split)
  - [Row-based version](row-split)
