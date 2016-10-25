## Introduction 

In March 2016, we released the first version of [XGBoost4J](http://dmlc.ml/2016/03/14/xgboost4j-portable-distributed-xgboost-in-spark-flink-and-dataflow.html), which is a set of packages providing Java/Scala interfaces of XGBoost and the integration with the prevalent JVM-based distributed data processing platforms, like Spark/Flink. 

The integration with Spark/Flink receives the tremendous positive feedbacks from the community. It enables users to build an unified pipeline, embedding the powerful XGBoost into the data processing system based on the prevalent frameworks like Spark. The following figure shows the general architecture of such a pipeline based on the first version of XGBoost4J, where the execution of data processing is based on the low-level Spark RDD abstraction.

![XGBoost4J Architecture](https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/unified_pipeline.png)

With the advancement of Spark, [DataFrame/Dataset-based APIs](http://spark.apache.org/docs/latest/sql-programming-guide.html) have been widely adopted. Unlike the low-level RDD abstraction, the interfaces of DataFrame/Dataset provide Spark with more information about data and computation and further enables Spark to use this extra information to perform extra optimizations. Users are calling for a integration between XGBoost and DataFrame/Dataset for gaining the performance improvement from both sides.

From Sep, we start working on a full integration between XGBoost and Spark's new DataFrame/Dataset-based abstraction. The following figure illustrates the new XGBoost4J architecture based on DataFrame/Dataset Abstraction.

![XGBoost4J New Architecture](https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/unified_pipeline_new.png)

## New Functionalities in XGBoost 

* DataFrame/Dataset-enabled interfaces

* Spark ML Pipeline Components
	* Estimator
	* Transformer 
	* Pipelining 

## An Example of Pipeline based on XGBoost and Spark


