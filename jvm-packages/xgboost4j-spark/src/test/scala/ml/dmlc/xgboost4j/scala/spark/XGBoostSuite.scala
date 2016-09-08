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

import java.io.File
import java.nio.file.Files

import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.util.Random

import org.apache.commons.logging.LogFactory
import org.apache.spark.mllib.linalg.{Vector => SparkVector, Vectors, DenseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FunSuite}

import ml.dmlc.xgboost4j.java.{Booster => JBooster, DMatrix => JDMatrix, XGBoostError}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, EvalTrait}

class XGBoostSuite extends FunSuite with BeforeAndAfter {

  private implicit var sc: SparkContext = null
  private val numWorkers = Runtime.getRuntime().availableProcessors()

  private class EvalError extends EvalTrait {

    val logger = LogFactory.getLog(classOf[EvalError])

    private[xgboost4j] var evalMetric: String = "custom_error"

    /**
     * get evaluate metric
     *
     * @return evalMetric
     */
    override def getMetric: String = evalMetric

    /**
     * evaluate with predicts and data
     *
     * @param predicts predictions as array
     * @param dmat     data matrix to evaluate
     * @return result of the metric
     */
    override def eval(predicts: Array[Array[Float]], dmat: DMatrix): Float = {
      var error: Float = 0f
      var labels: Array[Float] = null
      try {
        labels = dmat.getLabel
      } catch {
        case ex: XGBoostError =>
          logger.error(ex)
          return -1f
      }
      val nrow: Int = predicts.length
      for (i <- 0 until nrow) {
        if (labels(i) == 0.0 && predicts(i)(0) > 0) {
          error += 1
        } else if (labels(i) == 1.0 && predicts(i)(0) <= 0) {
          error += 1
        }
      }
      error / labels.length
    }
  }

  before {
    // build SparkContext
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("XGBoostSuite")
    sc = new SparkContext(sparkConf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  private def fromSVMStringToLabeledPoint(line: String): LabeledPoint = {
    val labelAndFeatures = line.split(" ")
    val label = labelAndFeatures(0).toInt
    val features = labelAndFeatures.tail
    val denseFeature = new Array[Double](129)
    for (feature <- features) {
      val idAndValue = feature.split(":")
      denseFeature(idAndValue(0).toInt) = idAndValue(1).toDouble
    }
    LabeledPoint(label, new DenseVector(denseFeature))
  }

  private def readFile(filePath: String): List[LabeledPoint] = {
    val file = Source.fromFile(new File(filePath))
    val sampleList = new ListBuffer[LabeledPoint]
    for (sample <- file.getLines()) {
      sampleList += fromSVMStringToLabeledPoint(sample)
    }
    sampleList.toList
  }

  private def buildTrainingRDD(sparkContext: Option[SparkContext] = None): RDD[LabeledPoint] = {
    val sampleList = readFile(getClass.getResource("/agaricus.txt.train").getFile)
    sparkContext.getOrElse(sc).parallelize(sampleList, numWorkers)
  }

  test("build RDD containing boosters with the specified worker number") {
    val trainingRDD = buildTrainingRDD()
    val testSet = readFile(getClass.getResource("/agaricus.txt.test").getFile).iterator
    import DataUtils._
    val testSetDMatrix = new DMatrix(new JDMatrix(testSet, null))
    val boosterRDD = XGBoost.buildDistributedBoosters(
      trainingRDD,
      List("eta" -> "1", "max_depth" -> "6", "silent" -> "0",
        "objective" -> "binary:logistic").toMap,
      new scala.collection.mutable.HashMap[String, String],
      numWorkers = 2, round = 5, eval = null, obj = null, useExternalMemory = false)
    val boosterCount = boosterRDD.count()
    assert(boosterCount === 2)
    val boosters = boosterRDD.collect()
    val eval = new EvalError()
    for (booster <- boosters) {
      // the threshold is 0.11 because it does not sync boosters with AllReduce
      val predicts = booster.predict(testSetDMatrix, outPutMargin = true)
      assert(eval.eval(predicts, testSetDMatrix) < 0.11)
    }
  }

  test("training with external memory cache") {
    sc.stop()
    sc = null
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("XGBoostSuite")
    val customSparkContext = new SparkContext(sparkConf)
    val eval = new EvalError()
    val trainingRDD = buildTrainingRDD(Some(customSparkContext))
    val testSet = readFile(getClass.getResource("/agaricus.txt.test").getFile).iterator
    import DataUtils._
    val testSetDMatrix = new DMatrix(new JDMatrix(testSet, null))
    val paramMap = List("eta" -> "1", "max_depth" -> "6", "silent" -> "0",
      "objective" -> "binary:logistic").toMap
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = numWorkers, useExternalMemory = true)
    assert(eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix) < 0.1)
    customSparkContext.stop()
    // clean
    val dir = new File(".")
    for (file <- dir.listFiles() if file.getName.startsWith("XGBoostSuite-0-dtrain_cache")) {
      file.delete()
    }
  }

  test("test with dense vectors containing missing value") {
    def buildDenseRDD(): RDD[LabeledPoint] = {
      val nrow = 100
      val ncol = 5
      val data0 = Array.ofDim[Double](nrow, ncol)
      // put random nums
      for (r <- 0 until nrow; c <- 0 until ncol) {
        data0(r)(c) = {
          if (c == ncol - 1) {
            -0.1
          } else {
            Random.nextDouble()
          }
        }
      }
      // create label
      val label0 = new Array[Double](nrow)
      for (i <- label0.indices) {
        label0(i) = Random.nextDouble()
      }
      val points = new ListBuffer[LabeledPoint]
      for (r <- 0 until nrow) {
        points += LabeledPoint(label0(r), Vectors.dense(data0(r)))
      }
      sc.parallelize(points)
    }
    val trainingRDD = buildDenseRDD().repartition(4)
    val testRDD = buildDenseRDD().repartition(4)
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "0",
      "objective" -> "binary:logistic").toMap
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    xgBoostModel.predict(testRDD.map(_.features.toDense), missingValue = -0.1f).collect()
  }

  test("test consistency of prediction functions with RDD") {
    val trainingRDD = buildTrainingRDD()
    val testSet = readFile(getClass.getResource("/agaricus.txt.test").getFile)
    val testRDD = sc.parallelize(testSet, numSlices = 1).map(_.features)
    val testCollection = testRDD.collect()
    for (i <- testSet.indices) {
      assert(testCollection(i).toDense.values.sameElements(testSet(i).features.toDense.values))
    }
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "0",
      "objective" -> "binary:logistic").toMap
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    val predRDD = xgBoostModel.predict(testRDD)
    val predResult1 = predRDD.collect()(0)
    import DataUtils._
    val predResult2 = xgBoostModel.booster.predict(new DMatrix(testSet.iterator))
    for (i <- predResult1.indices; j <- predResult1(i).indices) {
      assert(predResult1(i)(j) === predResult2(i)(j))
    }
  }

  test("test prediction functionality with empty partition") {
    def buildEmptyRDD(sparkContext: Option[SparkContext] = None): RDD[SparkVector] = {
      val sampleList = new ListBuffer[SparkVector]
      sparkContext.getOrElse(sc).parallelize(sampleList, numWorkers)
    }
    val trainingRDD = buildTrainingRDD()
    val testRDD = buildEmptyRDD()
    import DataUtils._
    val tempDir = Files.createTempDirectory("xgboosttest-")
    val tempFile = Files.createTempFile(tempDir, "", "")
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "0",
      "objective" -> "binary:logistic").toMap
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    println(xgBoostModel.predict(testRDD).collect().length === 0)
  }

  test("test model consistency after save and load") {
    val eval = new EvalError()
    val trainingRDD = buildTrainingRDD()
    val testSet = readFile(getClass.getResource("/agaricus.txt.test").getFile).iterator
    import DataUtils._
    val testSetDMatrix = new DMatrix(new JDMatrix(testSet, null))
    val tempDir = Files.createTempDirectory("xgboosttest-")
    val tempFile = Files.createTempFile(tempDir, "", "")
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "0",
      "objective" -> "binary:logistic").toMap
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

  test("nthread configuration must be equal to spark.task.cpus") {
    sc.stop()
    sc = null
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("XGBoostSuite").
      set("spark.task.cpus", "4")
    val customSparkContext = new SparkContext(sparkConf)
    // start another app
    val trainingRDD = buildTrainingRDD(Some(customSparkContext))
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "0",
      "objective" -> "binary:logistic", "nthread" -> 6).toMap
    intercept[IllegalArgumentException] {
      XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    }
    customSparkContext.stop()
  }

  test("kryoSerializer test") {
    sc.stop()
    sc = null
    val eval = new EvalError()
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("XGBoostSuite")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sparkConf.registerKryoClasses(Array(classOf[Booster]))
    val customSparkContext = new SparkContext(sparkConf)
    val trainingRDD = buildTrainingRDD(Some(customSparkContext))
    val testSet = readFile(getClass.getResource("/agaricus.txt.test").getFile).iterator
    import DataUtils._
    val testSetDMatrix = new DMatrix(new JDMatrix(testSet, null))
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "0",
      "objective" -> "binary:logistic").toMap
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    assert(eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix) < 0.1)
    customSparkContext.stop()
  }
}
