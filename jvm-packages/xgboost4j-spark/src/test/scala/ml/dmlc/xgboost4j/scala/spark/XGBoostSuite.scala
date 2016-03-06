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

import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.tools.reflect.Eval

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix, XGBoostError}
import ml.dmlc.xgboost4j.scala.{DMatrix, EvalTrait}
import org.apache.commons.logging.LogFactory
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class XGBoostSuite extends FunSuite with BeforeAndAfterAll {

  private var sc: SparkContext = null
  private val numWorker = 4

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

  override def beforeAll(): Unit = {
    // build SparkContext
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("XGBoostSuite")
    sc = new SparkContext(sparkConf)
  }

  override def afterAll(): Unit = {
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

  private def buildRDD(filePath: String): RDD[LabeledPoint] = {
    val sampleList = readFile(filePath)
    sc.parallelize(sampleList, numWorker)
  }

  private def buildTrainingRDD(): RDD[LabeledPoint] = {
    val trainRDD = buildRDD(getClass.getResource("/agaricus.txt.train").getFile)
    trainRDD
  }

  test("build RDD containing boosters") {
    val trainingRDD = buildTrainingRDD()
    val testSet = readFile(getClass.getResource("/agaricus.txt.test").getFile).iterator
    import DataUtils._
    val testSetDMatrix = new DMatrix(new JDMatrix(testSet, null))
    val boosterRDD = XGBoost.buildDistributedBoosters(
      trainingRDD,
      List("eta" -> "1", "max_depth" -> "2", "silent" -> "0",
        "objective" -> "binary:logistic").toMap,
      new scala.collection.mutable.HashMap[String, String],
      numWorker, 2, null, null)
    val boosterCount = boosterRDD.count()
    assert(boosterCount === numWorker)
    val boosters = boosterRDD.collect()
    for (booster <- boosters) {
      val predicts = booster.predict(testSetDMatrix, true)
      assert(new EvalError().eval(predicts, testSetDMatrix) < 0.1)
    }
  }
}
