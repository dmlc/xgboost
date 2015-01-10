Distributed XGBoost
======
This folder contains information about experimental version of distributed xgboost.

Build
=====
* In the root folder, run ```make```, this will give you xgboost, which uses rabit allreduce
  - this version of xgboost should be fault tolerant eventually
* Alterniatively, run ```make mpi```, this will give you xgboost-mpi
  - You will need to have MPI to build xgboost-mpi

Design Choice
=====
* XGBoost replies on [Rabit Library](https://github.com/tqchen/rabit)
* Rabit is an fault tolerant and portable allreduce library that provides Allreduce and Broadcast
* Since rabit is compatible with MPI, xgboost can be compiled using MPI backend

* How is the data distributed?
  - There are two solvers in distributed xgboost
  - Column-based solver split data by column, each node work on subset of columns, 
    it uses exactly the same algorithm as single node version.
  - Row-based solver split data by row, each node work on subset of rows,
    it uses an approximate histogram count algorithm, and will only examine subset of 
    potential split points as opposed to all split points.
  - Hadoop version can run on the existing hadoop platform,
    it use Rabit to submit jobs as map-reduce tasks.

Usage
====
* You will need a network filesystem, or copy data to local file system before running the code
* xgboost can be used together with submission script provided in Rabit on different possible types of job scheduler
* ***Note*** The distributed version is still multi-threading optimized.
    You should run one process per node that takes most available CPU,
    this will reduce the communication overhead and improve the performance.
   - One way to do that is limit mpi slot in each machine to be 1, or reserve nthread processors for each process.
* Examples:
  - [Column-based version](col-split)
  - [Row-based version](row-split)
