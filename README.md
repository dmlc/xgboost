xgboost: eXtreme Gradient Boosting 
======
An optimized general purpose gradient boosting (tree) library.

Contributors: https://github.com/tqchen/xgboost/graphs/contributors

Turorial and Documentation: https://github.com/tqchen/xgboost/wiki

Questions and Issues: [https://github.com/tqchen/xgboost/issues](https://github.com/tqchen/xgboost/issues?q=is%3Aissue+label%3Aquestion)

Notes on the Code: [Code Guide](src)

Features
======
* Sparse feature format:
  - Sparse feature format allows easy handling of missing values, and improve computation efficiency.
* Push the limit on single machine:
  - Efficient implementation that optimizes memory and computation.
* Speed: XGBoost is very fast
  - IN [demo/higgs/speedtest.py](demo/kaggle-higgs/speedtest.py), kaggle higgs data it is faster(on our machine 20 times faster using 4 threads) than sklearn.ensemble.GradientBoostingClassifier
* Layout of gradient boosting algorithm to support user defined objective
* Python interface, works with numpy and scipy.sparse matrix

Build
======
* Simply type make
* If your compiler does not come with OpenMP support, it will fire an warning telling you that the code will compile into single thread mode, and you will get single thread xgboost
* You may get a error: -lgomp is not found
  - You can type ```make no_omp=1```, this will get you single thread xgboost
  - Alternatively, you can upgrade your compiler to compile multi-thread version
* Possible way to build using Visual Studio (not tested):
  - In principle, you can put src/xgboost.cpp and src/io/io.cpp into the project, and build xgboost.
  - For python module, you need python/xgboost_wrapper.cpp and src/io/io.cpp to build a dll.

Version
======
* This version is named xgboost-unity, the code has been refactored from 0.2x to be cleaner and more flexibility
* This version of xgboost is not compatible with 0.2x, due to huge amount of changes in code structure
  - This means the model and buffer file of previous version can not be loaded in xgboost-unity
* For legacy 0.2x code, refer to [Here](https://github.com/tqchen/xgboost/releases/tag/v0.22)
* Change log in [CHANGES.md](CHANGES.md)

XGBoost in Graphlab Create
======
* XGBoost is adopted as part of boosted tree toolkit in Graphlab Create (GLC). Graphlab Create is a powerful toolkit that allows you to data manipulation, hyper-parameter search, visualization of big data in one framework. Try the Graphlab Create in [here](http://graphlab.com/products/create/quick-start-guide.html)
* [Nice blogpost](http://blog.graphlab.com/using-gradient-boosted-trees-to-predict-bike-sharing-demand) by Jay Gu using GLC boosted tree to solve kaggle bike sharing challenge
