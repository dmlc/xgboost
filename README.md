xgboost

eXtreme Gradient Boosting Library 
=======
Creater: Tianqi Chen

Features
=======
* Sparse feature format:
  - Sparse feature format allows easy handling of missing values, and improve computation efficiency.
* Push the limit on single machine:
  - Efficient implementation that optimizes memory and computation.
* Layout of gradient boosting algorithm to support generic tasks, see project wiki.

Planned key components
=======
* Gradient boosting models: 
    - regression tree (GBRT)
    - linear model/lasso
* Objectives to support tasks: 
    - regression
    - classification
    - ranking
    - matrix factorization
    - structured prediction
(3) OpenMP implementation

File extension convention: 
(1) .h are interface, utils and data structures, with detailed comment; 
(2) .cpp are implementations that will be compiled, with less comment; 
(3) .hpp are implementations that will be included by .cpp, with less comment

See also: https://github.com/tqchen/xgboost/wiki
