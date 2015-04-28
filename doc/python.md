XGBoost Python Module
====

This page will introduce XGBoost Python module, including:
* [Building and Import](#building-and-import)
* [Data Interface](#data-interface)
* [Setting Parameters](#setting-parameters)
* [Train Model](#training-model)
* [Early Stopping](#early-stopping)
* [Prediction](#prediction)

A [walk through python example](https://github.com/tqchen/xgboost/blob/master/demo/guide-python) for UCI Mushroom dataset is provided.

=
#### Install

To install XGBoost, you need to run `make` in the root directory of the project and then in the `wrappers` directory run

```shell
python setup.py install
```
Then import the module in Python as usual
```python
import xgboost as xgb
```

=
#### Data Interface
XGBoost python module is able to loading from libsvm txt format file, Numpy 2D array and xgboost binary buffer file. The data will be store in ```DMatrix``` object.

* To load libsvm text format file and XGBoost binary file into ```DMatrix```, the usage is like
```python
dtrain = xgb.DMatrix('train.svm.txt')
dtest = xgb.DMatrix('test.svm.buffer')
```
* To load numpy array into ```DMatrix```, the usage is like
```python
data = np.random.rand(5,10) # 5 entities, each contains 10 features
label = np.random.randint(2, size=5) # binary target
dtrain = xgb.DMatrix( data, label=label)
```
* Build ```DMatrix``` from ```scipy.sparse```
```python
csr = scipy.sparse.csr_matrix( (dat, (row,col)) )
dtrain = xgb.DMatrix( csr )
```
* Saving ```DMatrix``` into XGBoost binary file will make loading faster in next time. The usage is like:
```python
dtrain = xgb.DMatrix('train.svm.txt')
dtrain.save_binary("train.buffer")
```
* To handle missing value in ```DMatrix```, you can initialize the ```DMatrix``` like:
```python
dtrain = xgb.DMatrix( data, label=label, missing = -999.0)
```
* Weight can be set when needed, like
```python
w = np.random.rand(5,1)
dtrain = xgb.DMatrix( data, label=label, missing = -999.0, weight=w)
```


=
#### Setting Parameters
XGBoost use list of pair to save [parameters](parameter.md). Eg
* Booster parameters
```python
param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
plst = param.items()
plst += [('eval_metric', 'auc')] # Multiple evals can be handled in this way
plst += [('eval_metric', 'ams@0')]
```
* Specify validations set to watch performance
```python
evallist  = [(dtest,'eval'), (dtrain,'train')]
```

=
#### Training Model
With parameter list and data, you are able to train a model.
* Training
```python
num_round = 10
bst = xgb.train( plst, dtrain, num_round, evallist )
```
* Saving model
After training, you can save model and dump it out.
```python
bst.save_model('0001.model')
```
* Dump Model and Feature Map
You can dump the model to txt and review the meaning of model
```python
# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
bst.dump_model('dump.raw.txt','featmap.txt')
```
* Loading model
After you save your model, you can load model file at anytime by using
```python
bst = xgb.Booster({'nthread':4}) #init model
bst.load_model("model.bin") # load data
```
=
#### Early stopping

If you have a validation set, you can use early stopping to find the optimal number of boosting rounds. Early stopping requires at least one set in `evals`. If there's more than one, it will use the last.

`train(..., evals=evals, early_stopping_rounds=10)`

The model will train until the validation score stops improving. Validation error needs to decrease at least every `early_stopping_rounds` to continue training.

If early stopping occurs, the model will have two additional fields: `bst.best_score` and `bst.best_iteration`. Note that `train()` will return a model from the last iteration, not the best one.

This works with both metrics to minimize (RMSE, log loss, etc.) and to maximize (MAP, NDCG, AUC).

=
#### Prediction
After you training/loading a model and preparing the data, you can start to do prediction.
```python
data = np.random.rand(7,10) # 7 entities, each contains 10 features
dtest = xgb.DMatrix( data, missing = -999.0 )
ypred = bst.predict( xgmat )
```

If early stopping is enabled during training, you can predict with the best iteration.
```python
ypred = bst.predict(xgmat,ntree_limit=bst.best_iteration)
```
