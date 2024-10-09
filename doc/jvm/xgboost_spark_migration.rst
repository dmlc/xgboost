##########################################################
Migration Guide: How to migrate to XGBoost4j-Spark jvm 3.x
##########################################################

XGBoost4j-Spark jvm packages underwent significant modifications in version 3.0,
which may cause compatibility issues with existing user code.

This guide will walk you through the process of updating your code to ensure
it's compatible with XGBoost4j-Spark 3.0 and later versions.

************************
XGBoost4j Spark Packages
************************

XGBoost4j-Spark 3.0 has assembled xgboost4j package into xgboost4j-spark_2.12-3.0.0.jar, which means
you can now simply use `xgboost4j-spark` for your application.

* For CPU

  .. code-block:: xml

    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j-spark_${scala.binary.version}</artifactId>
        <version>3.0.0</version>
    </dependency>

* For GPU

  .. code-block:: xml

    <dependency>
        <groupId>ml.dmlc</groupId>
        <artifactId>xgboost4j-spark-gpu_${scala.binary.version}</artifactId>
        <version>3.0.0</version>
    </dependency>


When submitting the XGBoost application to the Spark cluster, you only need to specify the single `xgboost4j-spark` package.

* For CPU

  .. code-block:: bash

    spark-submit \
      --jars xgboost4j-spark_2.12-3.0.0.jar \
      ... \


* For GPU

  .. code-block:: bash

    spark-submit \
      --jars xgboost4j-spark-gpu_2.12-3.0.0.jar \
      ... \

***************
XGBoost Ranking
***************

Learning to rank using XGBoostRegressor has been replaced by a dedicated `XGBoostRanker`, which is specifically designed
to support ranking algorithms.

.. code-block:: scala

  // before xgboost4j-spark 3.0
  val regressor = new XGBoostRegressor().setObjective("rank:ndcg")

  // after xgboost4j-spark 3.0
  val ranker = new XGBoostRanker()

******************
Removed Parameters
******************

Starting from xgboost4j-spark 3.0, below parameters are removed.

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
