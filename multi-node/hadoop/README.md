Distributed XGBoost: Hadoop Version
====
*  The script in this fold shows an example of how to run distributed xgboost on hadoop platform.
*  It relies on [Rabit Library](https://github.com/tqchen/rabit) and Hadoop Streaming. 
*  Quick start: run ```bash run_binary_classification.sh <n_hadoop_workers> <path_in_HDFS>```
  - This is the hadoop version of binary classification example in the demo folder.
  - More info of the binary classification task can be refered to https://github.com/tqchen/xgboost/wiki/Binary-Classification.

Before you run the script
====
* Make sure you have set up the hadoop environment. Otherwise you should run single machine examples in the demo fold.
* Build: run ```bash build.sh``` in the root folder, it will automatically download rabit and build xgboost.
* Check whether the environment variable $HADOOP_HOME exists (e.g. run ```echo $HADOOP_HOME```). If not, plz set up hadoop-streaming.jar path in rabit_hadoop.py. 
 
How to Use
====
* Input data format: LIBSVM format. The example here uses generated data in demo/data folder.
* Put the training data in HDFS (hadoop distributed file system).
* Use rabit ```rabit_hadoop.py``` to submit training task to hadoop, and save the final model file.
* Get the final model file from HDFS, and locally do prediction as well as visualization of model.

XGBoost: Single machine verison VS Hadoop version
====
If you have used xgboost (single machine version) before, this section will show you how to run xgboost on hadoop with a slight modification on conf file.
* Hadoop version needs to set up how many slave nodes/machines/workers you would like to use at first. 
* IO: instead of reading and writing file locally, hadoop version use "stdin" to read training file and use "stdout" to store the final model file. Therefore, you should change the parameters "data" and "model_out" in conf file to ```data = stdin; model_out = stdout```.
* File cache: ```rabit_hadoop.py``` also provide several ways to cache necesary files, including binary file (xgboost), conf file, small size of dataset which used for eveluation during the training process, and so on.
  - Any file used in config file, excluding stdin, should be cached in the script. ```rabit_hadoop.py``` will automatically cache files in the command line. For example, ```rabit_hadoop.py -n 3 -i $hdfsPath/agaricus.txt.train -o $hdfsPath/mushroom.final.model $localPath/xgboost mushroom.hadoop.conf``` will cache "xgboost" and "mushroom.hadoop.conf".
  - You could also use "-f" to manually cache one or more files, like ```-f file1 -f file2``` or ```-f file1#file2``` (use "#" to spilt file names).
  - The local path of cached files in command is "./".
  - Since the cached files will be packaged and delivered to hadoop slave nodes, the cached file should not be large. For instance, trying to cache files of GB size may reduce the performance.
* Hadoop version also support evaluting each training round. You just need to modify parameters "eval_train" and "eval[test]" in conf file and cache the evaluation file.
* Hadoop version now can only save the final model.
* Predict locally. Althought the hadoop version supports training process, you should do prediction locally, just the same as single machine version.
* The hadoop version now can only save the final model. 
* More details of hadoop version can be referred to the usage of ```rabit_hadoop.py```.  

Notes
====
* The code has been tested on MapReduce 1 (MRv1), it should be ok to run on MapReduce 2 (MRv2, YARN).
* The code is multi-threaded, so you want to run one xgboost per node/worker, which means the parameter <n_workers> should be less than the number of slaves/workers. 

