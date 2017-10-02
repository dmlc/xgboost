Frequently Asked Questions
========================
This document contains frequently asked questions about xgboost.

How to tune parameters
----------------------
See [Parameter Tunning Guide](how_to/param_tuning.md)

Description on the model
------------------------
See [Introduction to Boosted Trees](model.md)


I have a big dataset
--------------------
XGBoost is designed to be memory efficient. Usually it can handle problems as long as the data fit into your memory
(This usually means millions of instances).
If you are running out of memory, checkout [external memory version](how_to/external_memory.md) or
[distributed version](../demo/distributed-training) of xgboost.


Running xgboost on Platform X (Hadoop/Yarn, Mesos)
--------------------------------------------------
The distributed version of XGBoost is designed to be portable to various environment.
Distributed XGBoost can be ported to any platform that supports [rabit](https://github.com/dmlc/rabit).
You can directly run xgboost on Yarn. In theory Mesos and other resource allocation engines can be easily supported as well.


Why not implement distributed xgboost on top of X (Spark, Hadoop)
-----------------------------------------------------------------
The first fact we need to know is going distributed does not necessarily solve all the problems.
Instead, it creates more problems such as more communication overhead and fault tolerance.
The ultimate question will still come back to how to push the limit of each computation node
and use less resources to complete the task (thus with less communication and chance of failure).

To achieve these, we decide to reuse the optimizations in the single node xgboost and build distributed version on top of it.
The demand of communication in machine learning is rather simple, in the sense that we can depend on a limited set of API (in our case rabit).
Such design allows us to reuse most of the code, while being portable to major platforms such as Hadoop/Yarn, MPI, SGE.
Most importantly, it pushes the limit of the computation resources we can use.


How can I port the model to my own system
-----------------------------------------
The model and data format of XGBoost is exchangeable,
which means the model trained by one language can be loaded in another.
This means you can train the model using R, while running prediction using
Java or C++, which are more common in production systems.
You can also train the model using distributed versions,
and load them in from Python to do some interactive analysis.


Do you support LambdaMART
-------------------------
Yes, xgboost implements LambdaMART. Checkout the objective section in [parameters](parameter.md)


How to deal with Missing Value
------------------------------
xgboost supports missing value by default.
In tree algorithms, branch directions for missing values are learned during training.
Note that the gblinear booster treats missing values as zeros.


Slightly different result between runs
--------------------------------------
This could happen, due to non-determinism in floating point summation order and multi-threading.
Though the general accuracy will usually remain the same.


Why do I see different results with sparse and dense data?
--------------------------------------------------------
"Sparse" elements are treated as if they were "missing" by the tree booster, and as zeros by the linear booster.
For tree models, it is important to use consistent data formats during training and scoring.