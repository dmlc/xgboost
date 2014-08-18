xgboost: eXtreme Gradient Boosting 
=======
An optimized general purpose gradient boosting (tree) library.

Contributors: https://github.com/tqchen/xgboost/graphs/contributors

Turorial and Documentation: https://github.com/tqchen/xgboost/wiki

Questions and Issues: [https://github.com/tqchen/xgboost/issues](https://github.com/tqchen/xgboost/issues?q=is%3Aissue+label%3Aquestion)

Features
=======
* Sparse feature format:
  - Sparse feature format allows easy handling of missing values, and improve computation efficiency.
* Push the limit on single machine:
  - Efficient implementation that optimizes memory and computation.
* Speed: XGBoost is very fast
  - IN [demo/higgs/speedtest.py](demo/kaggle-higgs/speedtest.py), kaggle higgs data it is faster(on our machine 20 times faster using 4 threads) than sklearn.ensemble.GradientBoostingClassifier
* Layout of gradient boosting algorithm to support user defined objective
* Python interface, works with numpy and scipy.sparse matrix

xgboost-unity
=======
* Experimental branch(not usable yet): refactor xgboost, cleaner code, more flexibility
* This version of xgboost is not compatible with 0.2x, due to huge amount of changes in code structure
  - This means the model and buffer file of previous version can not be loaded in xgboost-unity

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

