Distributed XGBoost: Hadoop Version
====
* Hadoop version: run ```bash run_binary_classification.sh <n_hadoop_workers> <path_in_HDFS>```
  - This is the hadoop version of binary classification example in the demo folder.

How to Use
====
* Check whether environment variable $HADOOP_HOME exists (e.g. run ```echo $HADOOP_HOME```). If not, plz set up hadoop-streaming.jar path in rabit_hadoop.py. 

Notes
====
* The code has been tested on MapReduce 1 (MRv1), it should be ok to run on MapReduce 2 (MRv2, YARN).
* The code is multi-threaded, so you want to run one xgboost per node/worker, which means the parameter <n_workers> should be less than the number of slaves/workers. 
* The hadoop version now can only save the final model and evaluate test data locally after the training process.

