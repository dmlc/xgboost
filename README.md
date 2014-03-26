xgboost: eXtreme Gradient Boosting 
=======
A General purpose gradient boosting (tree) library.

Creater: Tianqi Chen

Turorial and Documentation: https://github.com/tqchen/xgboost/wiki
 

Features
=======
* Sparse feature format:
  - Sparse feature format allows easy handling of missing values, and improve computation efficiency.
* Push the limit on single machine:
  - Efficient implementation that optimizes memory and computation.
* Layout of gradient boosting algorithm to support generic tasks, see project wiki.

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

File extension convention
=======
(1) .h are interface, utils and data structures, with detailed comment; 
(2) .cpp are implementations that will be compiled, with less comment; 
(3) .hpp are implementations that will be included by .cpp, with less comment

