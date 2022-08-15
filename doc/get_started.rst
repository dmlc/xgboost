########################
Get Started with XGBoost
########################

This is a quick start tutorial showing snippets for you to quickly try out XGBoost
on the demo dataset on a binary classification task.

********************************
Links to Other Helpful Resources
********************************
- See :doc:`Installation Guide </install>` on how to install XGBoost.
- See :doc:`Text Input Format </tutorials/input_format>` on using text format for specifying training/testing data.
- See :doc:`Tutorials </tutorials/index>` for tips and tutorials.
- See `Learning to use XGBoost by Examples <https://github.com/dmlc/xgboost/tree/master/demo>`_ for more code examples.

******
Python
******

.. code-block:: python

  from xgboost import XGBClassifier
  # read data
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  data = load_iris()
  X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
  # create model instance
  bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
  # fit model
  bst.fit(X_train, y_train)
  # make predictions
  preds = bst.predict(X_test)

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
