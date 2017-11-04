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

import scala.util.Random

import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.rabit.RabitTracker

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, Vectors, Vector => SparkVector}
import org.apache.spark.rdd.RDD
import org.scalatest.FunSuite

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

  test("build RDD containing boosters with the specified worker number") {
    val trainingRDD = sc.parallelize(Classification.train)
    val boosterRDD = XGBoost.buildDistributedBoosters(
      trainingRDD,
      List("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
        "objective" -> "binary:logistic").toMap,
      new java.util.HashMap[String, String](),
      numWorkers = 2, round = 5, eval = null, obj = null, useExternalMemory = true,
      missing = Float.NaN)
    val boosterCount = boosterRDD.count()
    assert(boosterCount === 2)
  }

  test("training with external memory cache") {
    import DataUtils._
    val eval = new EvalError()
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSetDMatrix = new DMatrix(Classification.test.iterator)
    val paramMap = List("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic").toMap
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = numWorkers, useExternalMemory = true)
    assert(eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix) < 0.1)
  }

  test("training with Scala-implemented Rabit tracker") {
    import DataUtils._
    val eval = new EvalError()
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSetDMatrix = new DMatrix(Classification.test.iterator)
    val paramMap = List("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic",
      "tracker_conf" -> TrackerConf(60 * 60 * 1000, "scala")).toMap
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = numWorkers)
    assert(eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix) < 0.1)
  }

  ignore("test with fast histo depthwise") {
    import DataUtils._
    val eval = new EvalError()
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSetDMatrix = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "gamma" -> "0.5", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "depthwise", "eval_metric" -> "error")
    // TODO: histogram algorithm seems to be very very sensitive to worker number
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = math.min(numWorkers, 2))
    assert(eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix) < 0.1)
  }

  ignore("test with fast histo lossguide") {
    import DataUtils._
    val eval = new EvalError()
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSetDMatrix = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "gamma" -> "0.5", "max_depth" -> "0", "silent" -> "1",
            "objective" -> "binary:logistic", "tree_method" -> "hist",
            "grow_policy" -> "lossguide", "max_leaves" -> "8", "eval_metric" -> "error")
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = math.min(numWorkers, 2))
    val x = eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix)
    assert(x < 0.1)
  }

  ignore("test with fast histo lossguide with max bin") {
    import DataUtils._
    val eval = new EvalError()
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSetDMatrix = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "gamma" -> "0.5", "max_depth" -> "0", "silent" -> "0",
            "objective" -> "binary:logistic", "tree_method" -> "hist",
            "grow_policy" -> "lossguide", "max_leaves" -> "8", "max_bin" -> "16",
            "eval_metric" -> "error")
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = math.min(numWorkers, 2))
    val x = eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix)
    assert(x < 0.1)
  }

  ignore("test with fast histo depthwidth with max depth") {
    import DataUtils._
    val eval = new EvalError()
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSetDMatrix = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "gamma" -> "0.5", "max_depth" -> "0", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "depthwise", "max_leaves" -> "8", "max_depth" -> "2",
      "eval_metric" -> "error")
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 10,
      nWorkers = math.min(numWorkers, 2))
    val x = eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix)
    assert(x < 0.1)
  }

  ignore("test with fast histo depthwidth with max depth and max bin") {
    import DataUtils._
    val eval = new EvalError()
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSetDMatrix = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "gamma" -> "0.5", "max_depth" -> "0", "silent" -> "0",
            "objective" -> "binary:logistic", "tree_method" -> "hist",
            "grow_policy" -> "depthwise", "max_depth" -> "2", "max_bin" -> "2",
            "eval_metric" -> "error")
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 10,
      nWorkers = math.min(numWorkers, 2))
    val x = eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix)
    assert(x < 0.1)
  }

  test("test with dense vectors containing missing value") {
    def buildDenseRDD(): RDD[MLLabeledPoint] = {
      val numRows = 100
      val numCols = 5

      val labeledPoints = (0 until numRows).map { _ =>
        val label = Random.nextDouble()
        val values = Array.tabulate[Double](numCols) { c =>
          if (c == numCols - 1) -0.1 else Random.nextDouble()
        }

        MLLabeledPoint(label, Vectors.dense(values))
      }

      sc.parallelize(labeledPoints)
    }

    val trainingRDD = buildDenseRDD().repartition(4)
    val testRDD = buildDenseRDD().repartition(4).map(_.features.asInstanceOf[DenseVector])
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic").toMap
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers,
      useExternalMemory = true)
    xgBoostModel.predict(testRDD, missingValue = -0.1f).collect()
  }

  test("test consistency of prediction functions with RDD") {
    import DataUtils._
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSet = Classification.test
    val testRDD = sc.parallelize(testSet, numSlices = 1).map(_.features)
    val testCollection = testRDD.collect()
    for (i <- testSet.indices) {
      assert(testCollection(i).toDense.values.sameElements(testSet(i).features.toDense.values))
    }
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic")
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    val predRDD = xgBoostModel.predict(testRDD)
    val predResult1 = predRDD.collect()
    assert(testRDD.count() === predResult1.length)
    val predResult2 = xgBoostModel.booster.predict(new DMatrix(testSet.iterator))
    for (i <- predResult1.indices; j <- predResult1(i).indices) {
      assert(predResult1(i)(j) === predResult2(i)(j))
    }
  }

  test("test eval functions with RDD") {
    import DataUtils._
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML).cache()
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic")
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5, nWorkers = numWorkers)
    // Nan Zhu: deprecate it for now
    // xgBoostModel.eval(trainingRDD, "eval1", iter = 5, useExternalCache = false)
    xgBoostModel.eval(trainingRDD, "eval2", evalFunc = new EvalError, useExternalCache = false)
  }

  test("test prediction functionality with empty partition") {
    import DataUtils._
    def buildEmptyRDD(sparkContext: Option[SparkContext] = None): RDD[SparkVector] = {
      sparkContext.getOrElse(sc).parallelize(List[SparkVector](), numWorkers)
    }
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testRDD = buildEmptyRDD()
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic").toMap
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    println(xgBoostModel.predict(testRDD).collect().length === 0)
  }

  test("test model consistency after save and load") {
    import DataUtils._
    val eval = new EvalError()
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSetDMatrix = new DMatrix(Classification.test.iterator)
    val tempDir = Files.createTempDirectory("xgboosttest-")
    val tempFile = Files.createTempFile(tempDir, "", "")
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic")
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    val evalResults = eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix)
    assert(evalResults < 0.1)
    xgBoostModel.saveModelAsHadoopFile(tempFile.toFile.getAbsolutePath)
    val loadedXGBooostModel = XGBoost.loadModelFromHadoopFile(tempFile.toFile.getAbsolutePath)
    val predicts = loadedXGBooostModel.booster.predict(testSetDMatrix, outPutMargin = true)
    val loadedEvalResults = eval.eval(predicts, testSetDMatrix)
    assert(loadedEvalResults == evalResults)
  }

  test("test save and load of different types of models") {
    import DataUtils._
    val tempDir = Files.createTempDirectory("xgboosttest-")
    val tempFile = Files.createTempFile(tempDir, "", "")
    var trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    var paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "reg:linear")
    // validate regression model
    var xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = numWorkers, useExternalMemory = false)
    xgBoostModel.setFeaturesCol("feature_col")
    xgBoostModel.setLabelCol("label_col")
    xgBoostModel.setPredictionCol("prediction_col")
    xgBoostModel.saveModelAsHadoopFile(tempFile.toFile.getAbsolutePath)
    var loadedXGBoostModel = XGBoost.loadModelFromHadoopFile(tempFile.toFile.getAbsolutePath)
    assert(loadedXGBoostModel.isInstanceOf[XGBoostRegressionModel])
    assert(loadedXGBoostModel.getFeaturesCol == "feature_col")
    assert(loadedXGBoostModel.getLabelCol == "label_col")
    assert(loadedXGBoostModel.getPredictionCol == "prediction_col")
    // classification model
    paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic")
    xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = numWorkers, useExternalMemory = false)
    xgBoostModel.asInstanceOf[XGBoostClassificationModel].setRawPredictionCol("raw_col")
    xgBoostModel.asInstanceOf[XGBoostClassificationModel].setThresholds(Array(0.5, 0.5))
    xgBoostModel.saveModelAsHadoopFile(tempFile.toFile.getAbsolutePath)
    loadedXGBoostModel = XGBoost.loadModelFromHadoopFile(tempFile.toFile.getAbsolutePath)
    assert(loadedXGBoostModel.isInstanceOf[XGBoostClassificationModel])
    assert(loadedXGBoostModel.asInstanceOf[XGBoostClassificationModel].getRawPredictionCol ==
      "raw_col")
    assert(loadedXGBoostModel.asInstanceOf[XGBoostClassificationModel].getThresholds.deep ==
      Array(0.5, 0.5).deep)
    assert(loadedXGBoostModel.getFeaturesCol == "features")
    assert(loadedXGBoostModel.getLabelCol == "label")
    assert(loadedXGBoostModel.getPredictionCol == "prediction")
    // (multiclass) classification model
    trainingRDD = sc.parallelize(MultiClassification.train).map(_.asML)
    paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "multi:softmax", "num_class" -> "6")
    xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = numWorkers, useExternalMemory = false)
    xgBoostModel.asInstanceOf[XGBoostClassificationModel].setRawPredictionCol("raw_col")
    xgBoostModel.asInstanceOf[XGBoostClassificationModel].setThresholds(
      Array(0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
    xgBoostModel.saveModelAsHadoopFile(tempFile.toFile.getAbsolutePath)
    loadedXGBoostModel = XGBoost.loadModelFromHadoopFile(tempFile.toFile.getAbsolutePath)
    assert(loadedXGBoostModel.isInstanceOf[XGBoostClassificationModel])
    assert(loadedXGBoostModel.asInstanceOf[XGBoostClassificationModel].getRawPredictionCol ==
      "raw_col")
    assert(loadedXGBoostModel.asInstanceOf[XGBoostClassificationModel].getThresholds.deep ==
      Array(0.5, 0.5, 0.5, 0.5, 0.5, 0.5).deep)
    assert(loadedXGBoostModel.asInstanceOf[XGBoostClassificationModel].numOfClasses == 6)
    assert(loadedXGBoostModel.getFeaturesCol == "features")
    assert(loadedXGBoostModel.getLabelCol == "label")
    assert(loadedXGBoostModel.getPredictionCol == "prediction")
  }

  test("test use groupData") {
    import DataUtils._
    val trainingRDD = sc.parallelize(Ranking.train0, numSlices = 1).map(_.asML)
    val trainGroupData: Seq[Seq[Int]] = Seq(Ranking.trainGroup0)
    val testRDD = sc.parallelize(Ranking.test, numSlices = 1).map(_.features)

    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "rank:pairwise", "eval_metric" -> "ndcg", "groupData" -> trainGroupData)

    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 2, nWorkers = 1)
    val predRDD = xgBoostModel.predict(testRDD)
    val predResult1: Array[Array[Float]] = predRDD.collect()
    assert(testRDD.count() === predResult1.length)

    val avgMetric = xgBoostModel.eval(trainingRDD, "test", iter = 0, groupData = trainGroupData)
    assert(avgMetric contains "ndcg")
    // If the labels were lost ndcg comes back as 1.0
    assert(avgMetric.split('=')(1).toFloat < 1F)
  }

  test("test use nested groupData") {
    import DataUtils._
    val trainingRDD0 = sc.parallelize(Ranking.train0, numSlices = 1)
    val trainingRDD1 = sc.parallelize(Ranking.train1, numSlices = 1)
    val trainingRDD = trainingRDD0.union(trainingRDD1).map(_.asML)

    val trainGroupData: Seq[Seq[Int]] = Seq(Ranking.trainGroup0, Ranking.trainGroup1)

    val testRDD = sc.parallelize(Ranking.test, numSlices = 1).map(_.features)

    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "rank:pairwise", "groupData" -> trainGroupData)

    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, nWorkers = 2)
    val predRDD = xgBoostModel.predict(testRDD)
    val predResult1: Array[Array[Float]] = predRDD.collect()
    assert(testRDD.count() === predResult1.length)
  }

  test("training with spark parallelism checks disabled") {
    import DataUtils._
    val eval = new EvalError()
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSetDMatrix = new DMatrix(Classification.test.iterator)
    val paramMap = List("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "timeout_request_workers" -> 0L).toMap
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = numWorkers)
    assert(eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix) < 0.1)
  }

  test("isClassificationTask correctly classifies supported objectives") {
    import org.scalatest.prop.TableDrivenPropertyChecks._

    val objectives = Table(
      ("isClassificationTask", "params"),
      (true, Map("obj_type" -> "classification")),
      (false, Map("obj_type" -> "regression")),
      (false, Map("objective" -> "rank:ndcg")),
      (false, Map("objective" -> "rank:pairwise")),
      (false, Map("objective" -> "rank:map")),
      (false, Map("objective" -> "count:poisson")),
      (true, Map("objective" -> "binary:logistic")),
      (true, Map("objective" -> "binary:logitraw")),
      (true, Map("objective" -> "multi:softmax")),
      (true, Map("objective" -> "multi:softprob")),
      (false, Map("objective" -> "reg:linear")),
      (false, Map("objective" -> "reg:logistic")),
      (false, Map("objective" -> "reg:gamma")),
      (false, Map("objective" -> "reg:tweedie")))
    forAll (objectives) { (isClassificationTask: Boolean, params: Map[String, String]) =>
      assert(XGBoost.isClassificationTask(params) == isClassificationTask)
    }
  }
}
