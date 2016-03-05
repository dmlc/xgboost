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

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class XGBoostSuite extends FunSuite with BeforeAndAfterAll {

  private var sc: SparkContext = null
  private val numWorker = 4

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

  private def buildRDD(filePath: String): RDD[LabeledPoint] = {
    val file = Source.fromFile(new File(filePath))
    val sampleList = new ListBuffer[LabeledPoint]
    for (sample <- file.getLines()) {
      sampleList += fromSVMStringToLabeledPoint(sample)
    }
    sc.parallelize(sampleList, numWorker)
  }

  private def buildTrainingAndTestRDD(): (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    val trainRDD = buildRDD(getClass.getResource("/agaricus.txt.train").getFile)
    val testRDD = buildRDD(getClass.getResource("/agaricus.txt.test").getFile)
    (trainRDD, testRDD)
  }

  test("build RDD containing boosters") {
    val (trainingRDD, testRDD) = buildTrainingAndTestRDD()
    val boosterRDD = XGBoost.buildDistributedBoosters(
      trainingRDD,
      Map[String, AnyRef](),
      numWorker, 4, null, null)
    val boosterCount = boosterRDD.count()
    assert(boosterCount === numWorker)
  }
}
