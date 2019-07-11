###################################
Distributed XGBoost with Kubernetes
###################################

Kubeflow community provides `XGBoost Operator <https://github.com/kubeflow/xgboost-operator>`_ to support distributed XGBoost training and batch prediction in a Kubernetes cluster. It provides an easy and efficient XGBoost model training and batch prediction in distributed fashion.

**********
How to use
**********
In order to run a XGBoost job in a Kubernetes cluster, carry out the following steps:

1. Install XGBoost Operator in Kubernetes.

   a. XGBoost Operator is designed to manage XGBoost jobs, including job scheduling, monitoring, pods and services recovery etc. Follow the `installation guide <https://github.com/kubeflow/xgboost-operator#installing-xgboost-operator>`_ to install XGBoost Operator.

2. Write application code to interface with the XGBoost operator.

   a. You'll need to furnish a few scripts to inteface with the XGBoost operator. Refer to the `Iris classification example <https://github.com/kubeflow/xgboost-operator/tree/master/config/samples/xgboost-dist>`_.
   b. Data reader/writer: you need to have your data source reader and writer based on the requirement. For example, if your data is stored in a Hive Table, you have to write your own code to read/write Hive table based on the ID of worker.
   c. Model persistence: in this example, model is stored in the OSS storage. If you want to store your model into Amazon S3, Google NFS or other storage, you'll need to specify the model reader and writer based on the requirement of storage system.

3. Configure the XGBoost job using a YAML file.

   a. YAML file is used to configure the computation resource and environment for your XGBoost job to run, e.g. the number of workers and masters. The template `YAML template <https://github.com/kubeflow/xgboost-operator/blob/master/config/samples/xgboost-dist/xgboostjob_v1alpha1_iris_train.yaml>`_ is provided for reference.

4. Submit XGBoost job to Kubernetes cluster.

   a. `Kubectl command <https://github.com/kubeflow/xgboost-operator#creating-a-xgboost-trainingprediction-job>`_ is used to submit a XGBoost job, and then you can monitor the job status.

****************
Work in progress
****************

- XGBoost Model serving
- Distributed data reader/writer from/to HDFS, HBase, Hive etc.
- Model persistence on Amazon S3, Google NFS etc.
