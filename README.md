xgboost: eXtreme Gradient Boosting 
=======
An optimized general purpose gradient boosting (tree) library.

Contributors: https://github.com/tqchen/xgboost/graphs/contributors

Turorial and Documentation: https://github.com/tqchen/xgboost/wiki
 
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

Supported key components
=======
* Gradient boosting models: 
    - regression tree (GBRT)
    - linear model/lasso
* Objectives to support tasks: 
    - regression
    - classification
* OpenMP implementation

Planned components
=======
* More objective to support tasks: 
    - ranking
    - matrix factorization
    - structured prediction

Build
======
* Simply type make
* If your compiler does not come with OpenMP support, it will fire an warning telling you that the code will compile into single thread mode, and you will get single thread xgboost
  - You may get a error: -lgomp is not found, you can remove -fopenmp flag in Makefile to get single thread xgboost, or upgrade your compiler to compile multi-thread version

File extension convention
=======
* .h are interface, utils and data structures, with detailed comment; 
* .cpp are implementations that will be compiled, with less comment; 
* .hpp are implementations that will be included by .cpp, with less comment
