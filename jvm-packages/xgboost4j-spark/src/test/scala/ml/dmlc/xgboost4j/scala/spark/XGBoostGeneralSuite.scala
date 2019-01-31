/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.spark

import java.nio.file.Files
import java.util.concurrent.LinkedBlockingDeque

import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.rabit.RabitTracker
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}
import org.apache.hadoop.fs.{FileSystem, Path}

import org.apache.spark.TaskContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql._
import org.scalatest.FunSuite
import scala.util.Random

import ml.dmlc.xgboost4j.java.Rabit

class XGBoostGeneralSuite extends FunSuite with PerTest {

  test("test Rabit allreduce to validate Scala-implemented Rabit tracker") {
    val vectorLength = 100
    val rdd = sc.parallelize(
      (1 to numWorkers * vectorLength).toArray.map { _ => Random.nextFloat() }, numWorkers).cache()

    val tracker = new RabitTracker(numWorkers)
    tracker.start(0)
    val trackerEnvs = tracker.getWorkerEnvs
    val collectedAllReduceResults = new LinkedBlockingDeque[Array[Float]]()

    val rawData = rdd.mapPartitions { iter =>
      Iterator(iter.toArray)
    }.collect()

    val maxVec = (0 until vectorLength).toArray.map { j =>
      (0 until numWorkers).toArray.map { i => rawData(i)(j) }.max
    }

    val allReduceResults = rdd.mapPartitions { iter =>
      Rabit.init(trackerEnvs)
      val arr = iter.toArray
      val results = Rabit.allReduce(arr, Rabit.OpType.MAX)
      Rabit.shutdown()
      Iterator(results)
    }.cache()

    val sparkThread = new Thread() {
      override def run(): Unit = {
        allReduceResults.foreachPartition(() => _)
        val byPartitionResults = allReduceResults.collect()
        assert(byPartitionResults(0).length == vectorLength)
        collectedAllReduceResults.put(byPartitionResults(0))
      }
    }
    sparkThread.start()
    assert(tracker.waitFor(0L) == 0)
    sparkThread.join()

    assert(collectedAllReduceResults.poll().sameElements(maxVec))
  }

  test("distributed training with the specified worker number") {
    val trainingRDD = sc.parallelize(Classification.train)
    val (booster, metrics) = XGBoost.trainDistributed(
      trainingRDD,
      List("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
        "objective" -> "binary:logistic", "num_round" -> 5, "num_workers" -> numWorkers,
        "custom_eval" -> null, "custom_obj" -> null, "use_external_memory" -> false,
        "missing" -> Float.NaN).toMap,
      hasGroup = false)
    assert(booster != null)
  }

  test("training with external memory cache") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "num_round" -> 5, "num_workers" -> numWorkers,
      "use_external_memory" -> true)
    val model = new XGBoostClassifier(paramMap).fit(training)
    assert(eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM) < 0.1)
  }


  test("training with Scala-implemented Rabit tracker") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "num_round" -> 5, "num_workers" -> numWorkers,
      "tracker_conf" -> TrackerConf(60 * 60 * 1000, "scala"))
    val model = new XGBoostClassifier(paramMap).fit(training)
    assert(eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM) < 0.1)
  }

  test("test with fast histo with monotone_constraints (lossguide)") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1",
      "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "tree_method" -> "hist", "grow_policy" -> "lossguide",
      "num_round" -> 5, "num_workers" -> numWorkers, "monotone_constraints" -> "(1, 0)")
    val model = new XGBoostClassifier(paramMap).fit(training)
    assert(eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM) < 0.1)
  }

  test("test with fast histo with interaction_constraints (lossguide)") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1",
      "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "tree_method" -> "hist", "grow_policy" -> "lossguide",
      "num_round" -> 5, "num_workers" -> numWorkers, "interaction_constraints" -> "[[1,2],[2,3,4]]")
    val model = new XGBoostClassifier(paramMap).fit(training)
    assert(eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM) < 0.1)
  }

  test("test with fast histo with monotone_constraints (depthwise)") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1",
      "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "tree_method" -> "hist", "grow_policy" -> "depthwise",
      "num_round" -> 5, "num_workers" -> numWorkers, "monotone_constraints" -> "(1, 0)")
    val model = new XGBoostClassifier(paramMap).fit(training)
    assert(eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM) < 0.1)
  }

  test("test with fast histo with interaction_constraints (depthwise)") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1",
      "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "tree_method" -> "hist", "grow_policy" -> "depthwise",
      "num_round" -> 5, "num_workers" -> numWorkers, "interaction_constraints" -> "[[1,2],[2,3,4]]")
    val model = new XGBoostClassifier(paramMap).fit(training)
    assert(eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM) < 0.1)
  }

  test("test with fast histo depthwise") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1",
      "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "tree_method" -> "hist", "grow_policy" -> "depthwise",
      "num_round" -> 5, "num_workers" -> numWorkers)
    val model = new XGBoostClassifier(paramMap).fit(training)
    assert(eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM) < 0.1)
  }

  test("test with fast histo depthwise with colsample_bytree") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1",
      "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "tree_method" -> "hist", "grow_policy" -> "depthwise",
      "num_round" -> 5, "num_workers" -> numWorkers, "colsample_bytree" -> 0.3)
    val model = new XGBoostClassifier(paramMap).fit(training)
    assert(eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM) < 0.1)
  }

  test("test with fast histo lossguide") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "gamma" -> "0.5", "max_depth" -> "0", "silent" -> "1",
      "objective" -> "binary:logistic", "tree_method" -> "hist", "grow_policy" -> "lossguide",
      "max_leaves" -> "8", "num_round" -> 5,
      "num_workers" -> numWorkers)
    val model = new XGBoostClassifier(paramMap).fit(training)
    val x = eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM)
    assert(x < 0.1)
  }

  test("test with fast histo lossguide with max bin") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "gamma" -> "0.5", "max_depth" -> "0", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "lossguide", "max_leaves" -> "8", "max_bin" -> "16",
      "eval_metric" -> "error", "num_round" -> 5, "num_workers" -> numWorkers)
    val model = new XGBoostClassifier(paramMap).fit(training)
    val x = eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM)
    assert(x < 0.1)
  }

  test("test with fast histo depthwidth with max depth") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "gamma" -> "0.5", "max_depth" -> "6", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "depthwise", "max_depth" -> "2",
      "eval_metric" -> "error", "num_round" -> 10, "num_workers" -> numWorkers)
    val model = new XGBoostClassifier(paramMap).fit(training)
    val x = eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM)
    assert(x < 0.1)
  }

  test("test with fast histo depthwidth with max depth and max bin") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "gamma" -> "0.5", "max_depth" -> "6", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "depthwise", "max_depth" -> "2", "max_bin" -> "2",
      "eval_metric" -> "error", "num_round" -> 10, "num_workers" -> numWorkers)
    val model = new XGBoostClassifier(paramMap).fit(training)
    val x = eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM)
    assert(x < 0.1)
  }

  test("dense vectors containing missing value") {
    def buildDenseDataFrame(): DataFrame = {
      val numRows = 100
      val numCols = 5

      val data = (0 until numRows).map { x =>
        val label = Random.nextInt(2)
        val values = Array.tabulate[Double](numCols) { c =>
          if (c == numCols - 1) -0.1 else Random.nextDouble
        }

        (label, Vectors.dense(values))
      }

      ss.createDataFrame(sc.parallelize(data.toList)).toDF("label", "features")
    }

    val denseDF = buildDenseDataFrame().repartition(4)
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic", "missing" -> -0.1f, "num_workers" -> numWorkers).toMap
    val model = new XGBoostClassifier(paramMap).fit(denseDF)
    model.transform(denseDF).collect()
  }

  test("training with spark parallelism checks disabled") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "timeout_request_workers" -> 0L,
      "num_round" -> 5, "num_workers" -> numWorkers)
    val model = new XGBoostClassifier(paramMap).fit(training)
    val x = eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM)
    assert(x < 0.1)
  }

  test("training with checkpoint boosters") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)

    val tmpPath = Files.createTempDirectory("model1").toAbsolutePath.toString
    val paramMap = Map("eta" -> "1", "max_depth" -> 2, "silent" -> "1",
      "objective" -> "binary:logistic", "checkpoint_path" -> tmpPath,
      "checkpoint_interval" -> 2, "num_workers" -> numWorkers)

    val prevModel = new XGBoostClassifier(paramMap ++ Seq("num_round" -> 5)).fit(training)
    def error(model: Booster): Float = eval.eval(
      model.predict(testDM, outPutMargin = true), testDM)

    // Check only one model is kept after training
    val files = FileSystem.get(sc.hadoopConfiguration).listStatus(new Path(tmpPath))
    assert(files.length == 1)
    assert(files.head.getPath.getName == "8.model")
    val tmpModel = SXGBoost.loadModel(s"$tmpPath/8.model")

    // Train next model based on prev model
    val nextModel = new XGBoostClassifier(paramMap ++ Seq("num_round" -> 8)).fit(training)
    assert(error(tmpModel) > error(prevModel._booster))
    assert(error(prevModel._booster) > error(nextModel._booster))
    assert(error(nextModel._booster) < 0.1)
  }

  test("repartitionForTrainingGroup with group data") {
    // test different splits to cover the corner cases.
    for (split <- 1 to 20) {
      val trainingRDD = sc.parallelize(Ranking.train, split)
      val traingGroupsRDD = XGBoost.repartitionForTrainingGroup(trainingRDD, 4)
      val trainingGroups: Array[Array[XGBLabeledPoint]] = traingGroupsRDD.collect()
      // check the the order of the groups with group id.
      // Ranking.train has 20 groups
      assert(trainingGroups.length == 20)

      // compare all points
      val allPoints = trainingGroups.sortBy(_(0).group).flatten
      assert(allPoints.length == Ranking.train.size)
      for (i <- 0 to Ranking.train.size - 1) {
        assert(allPoints(i).group == Ranking.train(i).group)
        assert(allPoints(i).label == Ranking.train(i).label)
        assert(allPoints(i).values.sameElements(Ranking.train(i).values))
      }
    }
  }

  test("repartitionForTrainingGroup with group data which has empty partition") {
    val trainingRDD = sc.parallelize(Ranking.train, 5).mapPartitions(it => {
      // make one partition empty for testing
      it.filter(_ => TaskContext.getPartitionId() != 3)
    })
    XGBoost.repartitionForTrainingGroup(trainingRDD, 4)
  }

  test("distributed training with group data") {
    val trainingRDD = sc.parallelize(Ranking.train, 5)
    val (booster, _) = XGBoost.trainDistributed(
      trainingRDD,
      List("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
        "objective" -> "rank:pairwise", "num_round" -> 5, "num_workers" -> numWorkers,
        "missing" -> Float.NaN, "use_external_memory" -> false).toMap,
      hasGroup = true)

    assert(booster != null)
  }

  test("training summary") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "num_round" -> 5, "nWorkers" -> numWorkers)

    val trainingDF = buildDataFrame(Classification.train)
    val xgb = new XGBoostClassifier(paramMap)
    val model = xgb.fit(trainingDF)

    assert(model.summary.trainObjectiveHistory.length === 5)
    assert(model.summary.validationObjectiveHistory.isEmpty)
  }

  test("train/test split") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "train_test_ratio" -> "0.5",
      "num_round" -> 5, "num_workers" -> numWorkers)
    val training = buildDataFrame(Classification.train)

    val xgb = new XGBoostClassifier(paramMap)
    val model = xgb.fit(training)
    assert(model.summary.validationObjectiveHistory.length === 1)
    assert(model.summary.validationObjectiveHistory(0)._1 === "test")
    assert(model.summary.validationObjectiveHistory(0)._2.length === 5)
    assert(model.summary.trainObjectiveHistory !== model.summary.validationObjectiveHistory(0))
  }

  test("train with multiple validation datasets (non-ranking)") {
    val training = buildDataFrame(Classification.train)
    val Array(train, eval1, eval2) = training.randomSplit(Array(0.6, 0.2, 0.2))
    val paramMap1 = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic",
      "num_round" -> 5, "num_workers" -> numWorkers)

    val xgb1 = new XGBoostClassifier(paramMap1).setEvalSets(Map("eval1" -> eval1, "eval2" -> eval2))
    val model1 = xgb1.fit(train)
    assert(model1.summary.validationObjectiveHistory.length === 2)
    assert(model1.summary.validationObjectiveHistory.map(_._1).toSet === Set("eval1", "eval2"))
    assert(model1.summary.validationObjectiveHistory(0)._2.length === 5)
    assert(model1.summary.validationObjectiveHistory(1)._2.length === 5)
    assert(model1.summary.trainObjectiveHistory !== model1.summary.validationObjectiveHistory(0))
    assert(model1.summary.trainObjectiveHistory !== model1.summary.validationObjectiveHistory(1))

    val paramMap2 = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic",
      "num_round" -> 5, "num_workers" -> numWorkers,
      "eval_sets" -> Map("eval1" -> eval1, "eval2" -> eval2))
    val xgb2 = new XGBoostClassifier(paramMap2)
    val model2 = xgb2.fit(train)
    assert(model2.summary.validationObjectiveHistory.length === 2)
    assert(model2.summary.validationObjectiveHistory.map(_._1).toSet === Set("eval1", "eval2"))
    assert(model2.summary.validationObjectiveHistory(0)._2.length === 5)
    assert(model2.summary.validationObjectiveHistory(1)._2.length === 5)
    assert(model2.summary.trainObjectiveHistory !== model2.summary.validationObjectiveHistory(0))
    assert(model2.summary.trainObjectiveHistory !== model2.summary.validationObjectiveHistory(1))
  }

  test("train with multiple validation datasets (ranking)") {
    val training = buildDataFrameWithGroup(Ranking.train, 5)
    val Array(train, eval1, eval2) = training.randomSplit(Array(0.6, 0.2, 0.2))
    val paramMap1 = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "rank:pairwise",
      "num_round" -> 5, "num_workers" -> numWorkers, "group_col" -> "group")
    val xgb1 = new XGBoostRegressor(paramMap1).setEvalSets(Map("eval1" -> eval1, "eval2" -> eval2))
    val model1 = xgb1.fit(train)
    assert(model1 != null)
    assert(model1.summary.validationObjectiveHistory.length === 2)
    assert(model1.summary.validationObjectiveHistory.map(_._1).toSet === Set("eval1", "eval2"))
    assert(model1.summary.validationObjectiveHistory(0)._2.length === 5)
    assert(model1.summary.validationObjectiveHistory(1)._2.length === 5)
    assert(model1.summary.trainObjectiveHistory !== model1.summary.validationObjectiveHistory(0))
    assert(model1.summary.trainObjectiveHistory !== model1.summary.validationObjectiveHistory(1))

    val paramMap2 = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "rank:pairwise",
      "num_round" -> 5, "num_workers" -> numWorkers, "group_col" -> "group",
      "eval_sets" -> Map("eval1" -> eval1, "eval2" -> eval2))
    val xgb2 = new XGBoostRegressor(paramMap2)
    val model2 = xgb2.fit(train)
    assert(model2 != null)
    assert(model2.summary.validationObjectiveHistory.length === 2)
    assert(model2.summary.validationObjectiveHistory.map(_._1).toSet === Set("eval1", "eval2"))
    assert(model2.summary.validationObjectiveHistory(0)._2.length === 5)
    assert(model2.summary.validationObjectiveHistory(1)._2.length === 5)
    assert(model2.summary.trainObjectiveHistory !== model2.summary.validationObjectiveHistory(0))
    assert(model2.summary.trainObjectiveHistory !== model2.summary.validationObjectiveHistory(1))
  }
}
