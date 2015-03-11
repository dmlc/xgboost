Distributed XGBoost: Hadoop Yarn Version
====
*  The script in this fold shows an example of how to run distributed xgboost on hadoop platform with YARN
*  It relies on [Rabit Library](https://github.com/tqchen/rabit) (Reliable Allreduce and Broadcast Interface) and Yarn. Rabit provides an interface to aggregate gradient values and split statistics, that allow xgboost to run reliably on hadoop. You do not need to care how to update model in each iteration, just use the script ```rabit_yarn.py```. For those who want to know how it exactly works, plz refer to the main page of [Rabit](https://github.com/tqchen/rabit).
*  Quick start: run ```bash run_mushroom.sh <n_hadoop_workers> <n_thread_per_worker> <path_in_HDFS>```
  - This is the hadoop version of binary classification example in the demo folder.
  - More info of the usage of xgboost can be refered to [wiki page](https://github.com/tqchen/xgboost/wiki)

Before you run the script
====
* Make sure you have set up the hadoop environment.  
  - Check variable $HADOOP_PREFIX exists (e.g. run ```echo $HADOOP_PREFIX```)
  - Compile xgboost with hdfs support by typing ```make hdfs=1```

How to Use
====
* Input data format: LIBSVM format. The example here uses generated data in demo/data folder.
* Put the training data in HDFS (hadoop distributed file system).
* Use rabit ```rabit_yarn.py``` to submit training task to yarn
* Get the final model file from HDFS, and locally do prediction as well as visualization of model.

Single machine vs Hadoop version
====
If you have used xgboost (single machine version) before, this section will show you how to run xgboost on hadoop with a slight modification on conf file.
* IO: instead of reading and writing file locally, we now use HDFS, put ```hdfs://``` prefix to the address of file you like to access
* File cache: ```rabit_yarn.py``` also provide several ways to cache necesary files, including binary file (xgboost), conf file
  - ```rabit_yarn.py``` will automatically cache files in the command line. For example, ```rabit_yarn.py -n 3 $localPath/xgboost mushroom.hadoop.conf``` will cache "xgboost" and "mushroom.hadoop.conf".
  - You could also use "-f" to manually cache one or more files, like ```-f file1 -f file2``` or ```-f file1#file2``` (use "#" to spilt file names).
  - The local path of cached files in command is "./".
  - Since the cached files will be packaged and delivered to hadoop slave nodes, the cached file should not be large.
* Hadoop version also support evaluting each training round. You just need to modify parameters "eval_train".
* More details of submission can be referred to the usage of ```rabit_yarn.py```.
* The model saved by hadoop version is compatible with single machine version.

Notes
====
* The code has been tested on YARN.
* The code is optimized with multi-threading, so you will want to run one xgboost per node/worker for best performance.
  - You will want to set <n_thread_per_worker> to be number of cores you have on each machine.
* It is also possible to submit job with hadoop streaming, however, YARN is highly recommended for efficiency reason
