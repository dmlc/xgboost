###############################
Distributed XGBoost on Kubernetes cluster
###############################

Distributed XGBoost includes model training and batch prediction. Beside the XGBoost over Spark, now, XGBoost job can run in Kubernetes cluster natively. 

Kubeflow community provides `XGBoost Operator <https://github.com/kubeflow/xgboost-operator>`_ to support distributed XGBoost training and batch prediction in a Kubernetes cluster. It provides an easy and efficient XGBoost model training and batch prediction in distributed way.  

**************
How it works ?
**************
In order to run the XGBoost job in a Kubernetes cluster, it requires following steps. 

- Step1: Install XGBoost Controller in Kubernetes
  - XGBoost Operator is designed to manage the XGBoost Job like job scheduling, monitoring, pods and services recovering etc. Thus, you can follow the `installment <https://github.com/kubeflow/xgboost-operator#installing-xgboost-operator>`_ guide to make XGBoost Job controller work at first.  

- Step2: Write your application and submit to XGBoost job controller

  - you can follow the `example <https://github.com/kubeflow/xgboost-operator/tree/master/config/samples/xgboost-dist>`_ to develop your own XGBoost application. 

  - Data reader/writer: you need to have your data source reader and writer based on the requirement. For example, if your data is stored in one Hive Table, you have to build your own code to read/write Hive table based on the ID of worker. 

  - Model persistence: in this example, model is stored in the OSS storage. If you want to store your model into Amazon S3, Google NFS or other storage. You also need to specific the model reader and writer based on the requirement of storage system.  

- Step3: Configure your Yaml file 

  - Yaml file is used to configure the computation resource and environment for your XGBoost job to run. Therefore, you can configure the Yaml file to specific your job runtime like the number of worker and master. The template `Yaml <https://github.com/kubeflow/xgboost-operator/blob/master/config/samples/xgboost-dist/xgboostjob_v1alpha1_iris_train.yaml>`_ is provided for you to refer.

- Step4: Submit job 

  - `Kubectl command <https://github.com/kubeflow/xgboost-operator#creating-a-xgboost-trainingprediction-job>` is used to submit a XGBoost job, and then you can monitor the job status as well. 

**************
Not Support
**************

- XGBoost Model serving 
- Distributed data reader/writer from/to HDFS, HBase, Hive etc.  
- Model persistence on Amazon S3, Google NFS etc. 
