Distributed XGBoost: Row Split Version
====
* You might be interested to checkout the [Hadoop example](../hadoop)
* Machine Rabit: run ```bash machine-row-rabit.sh <n-mpi-process>```
  - machine-col-rabit.sh starts xgboost job using rabit

How to Use
====
* First split the data by rows
* In the config, specify data file as containing a wildcard %d, where %d is the rank of the node, each node will load their part of data
* Enable ow split mode by ```dsplit=row```

Notes
====
* The code is multi-threaded, so you want to run one xgboost-mpi per node
* Row-based solver split data by row, each node work on subset of rows, it uses an approximate histogram count algorithm,
  and will only examine subset of potential split points as opposed to all split points.

