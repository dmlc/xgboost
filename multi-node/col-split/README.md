Distributed XGBoost: Column Split Version
====
* run ```bash run-mushroom.sh```

How to Use
====
* First split the data by column, 
* In the config, specify data file as containing a wildcard %d, where %d is the rank of the node, each node will load their part of data
* Enable column split mode by ```dsplit=col```

Notes
====
* The code is multi-threaded, so you want to run one xgboost-mpi per node
* The code will work correctly as long as union of each column subset is all the columns we are interested in.
  - The column subset can overlap with each other.
* It uses exactly the same algorithm as single node version, to examine all potential split points.
