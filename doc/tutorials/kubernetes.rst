###################################
Distributed XGBoost on Kubernetes
###################################

Distributed XGBoost training and batch prediction on `Kubernetes <https://kubernetes.io/>`_ are supported via `Kubeflow XGBoost Training Operator <https://github.com/kubeflow/training-operator>`_.

************
Instructions
************
In order to run a XGBoost job in a Kubernetes cluster, perform the following steps:

1. Install XGBoost Operator on the Kubernetes cluster.

   a. XGBoost Operator is designed to manage the scheduling and monitoring of XGBoost jobs. Follow `this installation guide <https://www.kubeflow.org/docs/components/training/xgboost/#installing-xgboost-operator>`_ to install XGBoost Operator.

2. Write application code that will be executed by the XGBoost Operator.

   a. To use XGBoost Operator, you'll have to write a couple of Python scripts that implement the distributed training logic for XGBoost. Please refer to the `Iris classification example <https://github.com/kubeflow/training-operator/tree/master/examples/xgboost/xgboost-dist>`_.
   b. Data reader/writer: you need to implement the data reader and writer based on the specific requirements of your chosen data source. For example, if your dataset is stored in a Hive table, you have to write the code to read from or write to the Hive table based on the index of the worker.
   c. Model persistence: in the `Iris classification example <https://github.com/kubeflow/training-operator/tree/master/examples/xgboost/xgboost-dist>`_, the model is stored in `Alibaba OSS <https://www.alibabacloud.com/product/oss>`_. If you want to store your model in other storages such as Amazon S3 or Google NFS, you'll need to implement the model persistence logic based on the requirements of the chosen storage system.

3. Configure the XGBoost job using a YAML file.

   a. YAML file is used to configure the computational resources and environment for your XGBoost job to run, e.g. the number of workers/masters and the number of CPU/GPUs. Please refer to this `YAML template <https://github.com/kubeflow/training-operator/blob/master/examples/xgboost/xgboost-dist/xgboostjob_v1alpha1_iris_train.yaml>`_ for an example.

4. Submit XGBoost job to a Kubernetes cluster.

   a. Use `kubectl <https://kubernetes.io/docs/reference/kubectl/overview/>`_ to submit a distributed XGBoost job as illustrated `here <https://www.kubeflow.org/docs/components/training/xgboost/#creating-a-xgboost-training-job>`_.

*******
Support
*******

Please submit an issue on `XGBoost Operator repo <https://github.com/kubeflow/training-operator/issues>`_ for any feature requests or problems.
