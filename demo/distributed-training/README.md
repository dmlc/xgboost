Distributed XGBoost Training
============================
This is an tutorial of Distributed XGBoost Training.
Currently xgboost supports distributed training via CLI program with the configuration file.
There is also plan push distributed python and other language bindings, please open an issue
if you are interested in contributing.

Build XGBoost with Distributed Filesystem Support
-------------------------------------------------
To use distributed xgboost, you only need to turn the options on to build
with distributed filesystems(HDFS or S3) in ```xgboost/make/config.mk```.


Step by Step Tutorial on AWS
----------------------------
Checkout [this tutorial](https://xgboost.readthedocs.org/en/latest/tutorials/aws_yarn.html) for running distributed xgboost.


Model Analysis
--------------
XGBoost is exchangeable across all bindings and platforms.
This means you can use python or R to analyze the learnt model and do prediction.
For example, you can use the [plot_model.ipynb](plot_model.ipynb) to visualize the learnt model.
