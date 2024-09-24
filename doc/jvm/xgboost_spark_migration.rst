########################################################
Migration Guide: How to migrate to XGBoost-Spark jvm 3.x
########################################################

XGBoost-Spark jvm packages underwent significant modifications in version 3.0,
which may cause compatibility issues with existing user code.

This guide will walk you through the process of updating your code to ensure
it's compatible with XGBoost-Spark 3.0 and later versions.

**********************
XGBoost Spark Packages
**********************

XGBoost-Spark 3.0 introduced a single uber package named xgboost-spark_2.12-3.0.0.jar, which bundles
both xgboost4j and xgboost4j-spark. This means you can now simply use `xgboost-spark`` for your application.

* For CPU

  .. code-block:: xml

    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost-spark_${scala.binary.version}</artifactId>
        <version>3.0.0</version>
    </dependency>

* For GPU

  .. code-block:: xml

    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost-spark-gpu_${scala.binary.version}</artifactId>
        <version>3.0.0</version>
    </dependency>


When submitting the XGBoost application to the Spark cluster, you only need to specify the single `xgboost-spark` package.

* For CPU

  .. code-block:: bash

    spark-submit \
      --jars xgboost-spark_2.12-3.0.0.jar \
      ... \


* For GPU

  .. code-block:: bash

    spark-submit \
      --jars xgboost-spark_2.12-3.0.0.jar \
      ... \

***************
XGBoost Ranking
***************

Learning to rank using XGBoostRegressor has been replaced by a dedicated `XGBoostRanker`, which is specifically designed
to support ranking algorithms.

.. code-block:: scala
  
  // before 3.0
  val regressor = new XGBoostRegressor().setObjective("rank:ndcg")

  // after 3.0
  val ranker = new XGBoostRanker()

******************************
XGBoost Constructor Parameters
******************************

XGBoost Spark now categorizes parameters into two groups: XGBoost-Spark parameters and XGBoost parameters.
When constructing an XGBoost estimator, only XGBoost-specific parameters are permitted. XGBoost-Spark specific 
parameters must be configured using the estimator's setter methods. It's worth noting that 
`XGBoost Parameters <https://xgboost.readthedocs.io/en/stable/parameter.html>`_
can be set both during construction and through the estimator's setter methods.

.. code-block:: scala

  // before 3.0
  val xgboost_paras = Map(
    "eta" -> "1",
    "max_depth" -> "6",
    "objective" -> "binary:logistic",
    "num_round" -> 5,
    "num_workers" -> 1,
    "features" -> "feature_column",
    "label" -> "label_column",
  )
  val classifier = new XGBoostClassifier(xgboost_paras)


  // after 3.0
  val xgboost_paras = Map(
    "eta" -> "1",
    "max_depth" -> "6",
    "objective" -> "binary:logistic",
    )
  val classifier = new XGBoostClassifier(xgboost_paras)
    .setNumRound(5)
    .setNumWorkers(1)
    .setFeaturesCol("feature_column")
    .setLabelCol("label_column")

  // Or you can use setter to set all parameters
  val classifier = new XGBoostClassifier()
    .setNumRound(5)
    .setNumWorkers(1)
    .setFeaturesCol("feature_column")
    .setLabelCol("label_column")
    .setEta(1)
    .setMaxDepth(6)
    .setObjective("binary:logistic")

******************
Removed Parameters
******************

Starting from 3.0, below parameters are removed.

- cacheTrainingSet

  If you wish to cache the training dataset, you have the option to implement caching
  in your code prior to fitting the data to an estimator.

  .. code-block:: scala
    
    val df = input.cache()
    val model = new XGBoostClassifier().fit(df)

- trainTestRatio

  The following method can be employed to do the evaluation.

  .. code-block:: scala
    
    val Array(train, eval) = trainDf.randomSplit(Array(0.7, 0.3))
    val classifier = new XGBoostClassifer().setEvalDataset(eval)
    val model = classifier.fit(train)

- tracker_conf

  The following method can be used to configure RabitTracker.

  .. code-block:: scala
    
    val classifier = new XGBoostClassifer()
      .setRabitTrackerTimeout(100)
      .setRabitTrackerHostIp("192.168.0.2")
      .setRabitTrackerPort(19203)

- rabitRingReduceThreshold
- rabitTimeout
- rabitConnectRetry
- singlePrecisionHistogram
- lambdaBias
- objectiveType
