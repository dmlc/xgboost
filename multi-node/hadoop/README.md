Distributed XGBoost: Hadoop Version
====
* Hadoop version: run ```bash run_binary_classification.sh <n_hadoop_workers> <n_thread_per_worker> <path_in_HDFS>```
  - This is the hadoop version of binary classification example in the demo folder.

How to Use
====
* Check whether environment variable $HADOOP_HOME exists (e.g. run ```echo $HADOOP_HOME```). If not, plz set up hadoop-streaming.jar path in rabit_hadoop.py. 

Notes
====
* The code has been tested on MapReduce 1 (MRv1) and YARN, it recommended run on MapReduce 2 (MRv2, YARN).
* The code is multi-threaded, so you want to run one xgboost per node/worker, which means you want to set <n_thread_per_worker> to be number of cores you have on each machine.
  - You will need YARN to set specify number of cores of each worker
* The hadoop version save the final model into HDFS
