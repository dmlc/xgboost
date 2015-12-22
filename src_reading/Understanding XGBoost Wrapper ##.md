## Understanding XGBoost Wrapper ##

Booster is the main class which suports the operations called by the external language bindings. Booster is implemented in `wrapper/xgboost_wrapper.cpp`. The Booster class inherited from BoostLearner, which is the class making training and prediction based on given loss function. 

`BoostLearner::InitModel` is called by `XGBoostWrapper::CheckInitModel()`. In `BoostLearner::InitModel`, the program initializes the trainer, gradient boosting model as well as the  object function. 

`Trainer` is the executable entity which synchronizes the parameters across the nodes, e.g. 

> // run allreduce on num_feature to find the maximum value                                               
rabit::Allreduce\<rabit::op::Max>(&num_feature, 1);    

The above lines calls rabit library to find hte maximum number of features across the nodes. 


The next question is how the distributed trainer of the boosting tree communicates with each other? XGBoost relies on rabit library for communication. In the last example, the program calls rabit:Allreduce to synchronize the number of features. 

### TODO: analyze the implementation of allreduce###

