xgboost: eXtreme Gradient Boosting 
======
An optimized general purpose gradient boosting library. The library is parallelized using OpenMP. It implements machine learning algorithm under gradient boosting framework, including generalized linear model and gradient boosted regression tree. 

Contributors: https://github.com/tqchen/xgboost/graphs/contributors

Turorial and Documentation: https://github.com/tqchen/xgboost/wiki

Questions and Issues: [https://github.com/tqchen/xgboost/issues](https://github.com/tqchen/xgboost/issues?q=is%3Aissue+label%3Aquestion)

Examples Code: [Learning to use xgboost by examples](demo)

Notes on the Code: [Code Guide](src)

What's New
=====
* See the updated [demo folder](demo) for feature walkthrough
* Thanks to Tong He, the new [R package](R-package) is available

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
=====
* Run ```bash build.sh``` (you can also type make)
* If your compiler does not come with OpenMP support, it will fire an warning telling you that the code will compile into single thread mode, and you will get single thread xgboost
* You may get a error: -lgomp is not found
  - You can type ```make no_omp=1```, this will get you single thread xgboost
  - Alternatively, you can upgrade your compiler to compile multi-thread version
* Windows(VS 2010): see [windows](windows) folder
  - In principle, you put all the cpp files in the Makefile to the project, and build

Version
======
* This version xgboost-0.3, the code has been refactored from 0.2x to be cleaner and more flexibility
* This version of xgboost is not compatible with 0.2x, due to huge amount of changes in code structure
  - This means the model and buffer file of previous version can not be loaded in xgboost-3.0
* For legacy 0.2x code, refer to [Here](https://github.com/tqchen/xgboost/releases/tag/v0.22)
* Change log in [CHANGES.md](CHANGES.md)

XGBoost in Graphlab Create
======
* XGBoost is adopted as part of boosted tree toolkit in Graphlab Create (GLC). Graphlab Create is a powerful python toolkit that allows you to data manipulation, graph processing, hyper-parameter search, and visualization of TeraBytes scale data in one framework. Try the Graphlab Create in http://graphlab.com/products/create/quick-start-guide.html
* Nice blogpost by Jay Gu using GLC boosted tree to solve kaggle bike sharing challenge: http://blog.graphlab.com/using-gradient-boosted-trees-to-predict-bike-sharing-demand
