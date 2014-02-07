xgboost
=======
Creater: Tianqi Chen: tianqi.tchen AT gmail

General Purpose Gradient Boosting Library

Intention: A stand-alone efficient library to do machine learning in functional space

Planned key components (TODO):

(1) Gradient boosting models: 
    - regression tree
    - linear model/lasso
(2) Objectives to support tasks: 
    - regression
    - classification
    - ranking
    - matrix factorization
    - structured prediction
(3) OpenMP support for parallelization(optional)

File extension convention: 
(1) .h are interface, utils anddata structures, with detailed comment; 
(2) .cpp are implementations that will be compiled, with less comment; 
(3) .hpp are implementations that will be included by .cpp, with less comment
