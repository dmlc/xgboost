Distributed XGBoost: Row Split Version
====
* Mushroom: run ```bash mushroom-row.sh <n-mpi-process>```
* Machine: run ```bash machine-row.sh <n-mpi-process>```

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
* ```colsample_bytree``` is not enabled in row split mode so far
