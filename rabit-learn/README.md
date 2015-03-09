Rabit-Learn
====
This folder contains implementation of distributed machine learning algorithm using rabit.
It also contain links to the Machine Learning packages that uses rabit.

* Contribution of toolkits, examples, benchmarks is more than welcomed!


Toolkits
====
* [KMeans Clustering](kmeans)
* [Linear and Logistic Regression](linear)  
* [XGBoost: eXtreme Gradient Boosting](https://github.com/tqchen/xgboost/tree/master/multi-node)
  - xgboost is a very fast boosted tree(also known as GBDT) library, that can run more than
    10 times faster than existing packages
  - Rabit carries xgboost to distributed enviroment, inheritating all the benefits of xgboost
    single node version, and scale it to even larger problems
