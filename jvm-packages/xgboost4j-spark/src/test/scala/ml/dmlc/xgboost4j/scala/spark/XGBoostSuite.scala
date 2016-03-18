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

import org.apache.commons.logging.LogFactory
import org.apache.spark.mllib.linalg.DenseVector
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
      List("eta" -> "1", "max_depth" -> "2", "silent" -> "0",
        "objective" -> "binary:logistic").toMap,
      new scala.collection.mutable.HashMap[String, String],
      numWorkers = 2, round = 5, null, null)
    val boosterCount = boosterRDD.count()
    assert(boosterCount === 2)
    val boosters = boosterRDD.collect()
    for (booster <- boosters) {
      val predicts = booster.predict(testSetDMatrix, true)
      assert(new EvalError().eval(predicts, testSetDMatrix) < 0.17)
    }
  }


  test("save and load model") {
    val eval = new EvalError()
    val trainingRDD = buildTrainingRDD()
    val testSet = readFile(getClass.getResource("/agaricus.txt.test").getFile).iterator
    import DataUtils._
    val testSetDMatrix = new DMatrix(new JDMatrix(testSet, null))
    val tempDir = Files.createTempDirectory("xgboosttest-")
    val tempFile = Files.createTempFile(tempDir, "", "")
    val paramMap = List("eta" -> "1", "max_depth" -> "2", "silent" -> "0",
      "objective" -> "binary:logistic").toMap
    val xgBoostModel = XGBoost.train(trainingRDD, paramMap, 5, numWorkers)
    assert(eval.eval(xgBoostModel.predict(testSetDMatrix), testSetDMatrix) < 0.1)
    xgBoostModel.saveModelAsHadoopFile(tempFile.toFile.getAbsolutePath)
    val loadedXGBooostModel = XGBoost.loadModelFromHadoopFile(tempFile.toFile.getAbsolutePath)
    val predicts = loadedXGBooostModel.predict(testSetDMatrix)
    assert(eval.eval(predicts, testSetDMatrix) < 0.1)
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
      XGBoost.train(trainingRDD, paramMap, 5, numWorkers)
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
    val xgBoostModel = XGBoost.train(trainingRDD, paramMap, 5, numWorkers)
    assert(eval.eval(xgBoostModel.predict(testSetDMatrix), testSetDMatrix) < 0.1)
    customSparkContext.stop()
  }
}
