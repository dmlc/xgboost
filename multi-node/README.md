Distributed XGBoost
======
Distributed XGBoost is now part of [Wormhole](https://github.com/dmlc/wormhole).
Checkout this [Link](https://github.com/dmlc/wormhole/tree/master/learn/xgboost) for usage examples, build and job submissions.
* The distributed version is built on Rabit:[Reliable Allreduce and Broadcast Library](https://github.com/dmlc/rabit)
  - Rabit is a portable library that provides fault-tolerance for Allreduce calls for distributed machine learning  
  - This makes xgboost portable and fault-tolerant against node failures

Notes
====
* Rabit handles all the fault tolerant and communications efficiently, we only use platform specific command to start programs
  - The Hadoop version does not rely on Mapreduce to do iterations
  - You can expect xgboost not suffering the drawbacks of iterative MapReduce program
* The design choice was made because Allreduce is very natural and efficient for distributed tree building
  - In current version of xgboost, the distributed version is only adds several lines of Allreduce synchronization code
* The multi-threading nature of xgboost is inheritated in distributed mode
  - This means xgboost efficiently use all the threads in one machine, and communicates only between machines
  - Remember to run on xgboost process per machine and this will give you maximum speedup
* For more information about rabit and how it works, see the [Rabit's Tutorial](https://github.com/dmlc/rabit/tree/master/guide)

Solvers
=====
* Column-based solver split data by column, each node work on subset of columns, 
  it uses exactly the same algorithm as single node version.
* Row-based solver split data by row, each node work on subset of rows,
  it uses an approximate histogram count algorithm, and will only examine subset of 
  potential split points as opposed to all split points.
  - This is the mode used by current hadoop version, since usually data was stored by rows in many industry system
