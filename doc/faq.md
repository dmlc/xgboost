Frequent Asked Questions
========================
This document contains the frequent asked question to xgboost.

How to tune parameters
----------------------
See [Parameter Tunning Guide](param_tuning.md)

Description on the model
------------------------
See [Introduction to Boosted Trees](model.md)


I have a big dataset
--------------------
XGBoost is designed to be memory efficient. Usually it could handle problems as long as the data fit into your memory
(This usually means millions of instances).
If you are running out of memory, checkout [external memory version](external_memory.md) or
[distributed version](https://github.com/dmlc/wormhole/tree/master/learn/xgboost) of xgboost.


Running xgboost on Platform X (Hadoop/Yarn, Mesos)
--------------------------------------------------
The distributed version of XGBoost is designed to be portable to various environment.
Distributed XGBoost can be ported to any platform that supports [rabit](https://github.com/dmlc/rabit).
You can directly run xgboost on Yarn. In theory Mesos and other resource allocation engine can be easily supported as well.


Why not implement distributed xgboost on top of X (Spark, Hadoop)
-----------------------------------------------------------------
The first fact we need to know is going distributed does not necessarily solve all the problems.
Instead, it creates more problems such as more communication over head and fault tolerance.
The ultimate question will still come back into how to push the limit of each computation node
and use less resources to complete the task (thus with less communication and chance of failure).

To achieve these, we decide to reuse the optimizations in the single node xgboost and build distributed version on top of it.
The demand of communication in machine learning is rather simple, in a sense that we can depend on a limited set of API (in our case rabit).
Such design allows us to reuse most of the code, and being portable to major platforms such as Hadoop/Yarn, MPI, SGE.
Most importantly, pushs the limit of the computation resources we can use.


How can I port the model to my own system
-----------------------------------------
The model and data format of XGBoost is exchangable.
Which means the model trained by one langauge can be loaded in another.
This means you can train the model using R, while running prediction using
Java or C++, which are more common in production system.
You can also train the model using distributed version,
and load them in from python to do some interactive analysis.


Do you support LambdaMART
-------------------------
Yes, xgboost implements LambdaMART. Checkout the objective section in [parameters](parameter.md)


How to deal with Missing Value
------------------------------
xgboost support missing value by default


Slightly different result between runs
--------------------------------------
This could happen, due to non-determinism in floating point summation order and multi-threading.
Though the general accuracy will usually remain the same.
