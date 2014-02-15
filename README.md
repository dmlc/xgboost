xgboost: A Gradient Boosting Library
=======
Creater: Tianqi Chen: tianqi.tchen AT gmail

General Purpose Gradient Boosting Library

Goal: A stand-alone efficient library to do learning via boosting in functional space

Features:
* Sparse feature format, handling of missing features. This allows efficient categorical feature encoding as indicators. The speed of booster only depends on number of existing features.
* Layout of gradient boosting algorithm to support generic tasks, see project wiki.


Planned key components:

* Gradient boosting models: 
    - regression tree (GBRT)
    - linear model/lasso
* Objectives to support tasks: 
    - regression
    - classification
    - ranking
    - matrix factorization
    - structured prediction
(3) OpenMP implementation(optional)

File extension convention: 
(1) .h are interface, utils and data structures, with detailed comment; 
(2) .cpp are implementations that will be compiled, with less comment; 
(3) .hpp are implementations that will be included by .cpp, with less comment

See also: https://github.com/tqchen/xgboost/wiki
