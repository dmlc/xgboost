Distributed XGBoost
======
This folder contains information about experimental version of distributed xgboost.

Build
=====
* You will need to have MPI
* In the root folder, run ```make mpi```, this will give you xgboost-mpi

Design Choice
=====
* Does distributed xgboost reply on MPI?
  - Yes, but the dependency is isolated in [sync](../src/sync/sync.h) module
  - Specificially, xgboost reply on MPI protocol that provide Broadcast and AllReduce,
     if there are platform/framework that implements these protocol, xgboost should naturally extends to these platform
* How is the data distributed?
  - There are two solvers in distributed xgboost
  - Column-based solver split data by column, each node work on subset of columns, 
    it uses exactly the same algorithm as single node version.
  - Row-based solver split data by row, each node work on subset of rows,
    it uses an approximate histogram count algorithm, and will only examine subset of 
    potential split points as opposed to all split points.
* How to run the distributed version
  - The current code run in MPI enviroment, you will need to have a network filesystem,
    or copy data to local file system before running the code
  - The distributed version is still multi-threading optimized.
    You should run one xgboost-mpi per node that takes most available CPU,
    this will reduce the communication overhead and improve the performance.
  - One way to do that is limit mpi slot in each machine to be 1, or reserve nthread processors for each process.
  
Usage
====
* [Column-based version](col-split)
* [Row-based version](row-split)
