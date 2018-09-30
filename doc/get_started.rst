########################
Get Started with XGBoost
########################

This is a quick start tutorial showing snippets for you to quickly try out XGBoost
on the demo dataset on a binary classification task.

********************************
Links to Other Helpful Resources
********************************
- See :doc:`Installation Guide </build>` on how to install XGBoost.
- See :doc:`Text Input Format </tutorials/input_format>` on using text format for specifying training/testing data.
- See :doc:`Tutorials </tutorials/index>` for tips and tutorials.
- See `Learning to use XGBoost by Examples <https://github.com/dmlc/xgboost/tree/master/demo>`_ for more code examples.

******
Python
******

.. code-block:: python

  import xgboost as xgb
  # read in data
  dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
  dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
  # specify parameters via map
  param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
  num_round = 2
  bst = xgb.train(param, dtrain, num_round)
  # make prediction
  preds = bst.predict(dtest)

***
R
***

.. code-block:: R

  # load data
  data(agaricus.train, package='xgboost')
  data(agaricus.test, package='xgboost')
  train <- agaricus.train
  test <- agaricus.test
  # fit model
  bst <- xgboost(data = train$data, label = train$label, max.depth = 2, eta = 1, nrounds = 2,
                 nthread = 2, objective = "binary:logistic")
  # predict
  pred <- predict(bst, test$data)

*****
Julia
*****

.. code-block:: julia

  using XGBoost
  # read data
  train_X, train_Y = readlibsvm("demo/data/agaricus.txt.train", (6513, 126))
  test_X, test_Y = readlibsvm("demo/data/agaricus.txt.test", (1611, 126))
  # fit model
  num_round = 2
  bst = xgboost(train_X, num_round, label=train_Y, eta=1, max_depth=2)
  # predict
  pred = predict(bst, test_X)

*****
Scala
*****

.. code-block:: scala

  import ml.dmlc.xgboost4j.scala.DMatrix
  import ml.dmlc.xgboost4j.scala.XGBoost
  
  object XGBoostScalaExample {
    def main(args: Array[String]) {
      // read trainining data, available at xgboost/demo/data
      val trainData =
        new DMatrix("/path/to/agaricus.txt.train")
      // define parameters
      val paramMap = List(
        "eta" -> 0.1,
        "max_depth" -> 2,
        "objective" -> "binary:logistic").toMap
      // number of iterations
      val round = 2
      // train the model
      val model = XGBoost.train(trainData, paramMap, round)
      // run prediction
      val predTrain = model.predict(trainData)
      // save model to the file.
      model.saveModel("/local/path/to/model")
    }
  }
