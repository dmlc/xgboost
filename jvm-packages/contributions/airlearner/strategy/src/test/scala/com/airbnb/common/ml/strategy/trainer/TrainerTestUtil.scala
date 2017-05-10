package com.airbnb.common.ml.strategy.trainer

import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.HiveContext

import com.airbnb.common.ml.strategy.config.TrainingOptions
import com.airbnb.common.ml.strategy.data.BaseBinarySample
import com.airbnb.common.ml.util.{ScalaLogging, TestUtil}


object TrainerTestUtil extends ScalaLogging {

  val key = "10"
  def getOptions(learning: String = "leadDays"): TrainingOptions = {
    TrainingOptions(
      // trueLowerBound, falseUpperBound,
      0.98, 1.0,
      // falseLowerBound trueUpperBound
      0.65, 1.03,
      // r0
      0.1, 0.2, 0.95,
      // numEpochs: Int, miniBatchSize: Int,
      100, 50,
      0.5,
      0, 0.0,
      // min: Array[Double], max: Array[Double],
      List(0.1, 0), List(3, 3),
      List(),
      // dsEval: String, learningRateType: String
      "2016-08-31", learning)
  }

  def getOptionsTanhv2(learning: String = "leadDays"): TrainingOptions = {
    TrainingOptions(
      // trueLowerBound, falseUpperBound,
      0.98, 1.0,
      // falseLowerBound trueUpperBound
      0.65, 1.03,
      // r0
      0.1, 0.2, 0.95,
      // numEpochs: Int, miniBatchSize: Int,
      100, 50,
      0.5,
      0, 0.5,
      // min: Array[Double], max: Array[Double],
      List(0.05, 4, -24),
      List(0.35, 25.0, -1.0),
      List(),
      // dsEval: String, learningRateType: String
      "2016-08-31", learning)
  }

  // fixed key for test on single key case.
  def parseKey(cols: Array[String]): String = {
    key
  }

  def parseBaseData(cols: Array[String]): BaseBinarySample = {
    //  label: Boolean,
    //  x: Double,
    //  pivotValue: Double,
    //  observedValue: Double
    BaseBinarySample(
      cols(0).toBoolean,
      cols(1).toDouble,
      cols(2).toDouble,
      cols(3).toDouble,
      Option(cols(4).toDouble))
  }

  def testSearchAllBestOptions(
       training: String,
       eval: String,
       hc: HiveContext,
       sc: SparkContext,
       trainer: BinaryRegressionTrainer[BaseBinarySample]
  ): Unit = {
    val options = TrainerTestUtil.getOptionsTanhv2()
    val trainingExamples = TestUtil.parseCSVToRDD(
      training, TrainerTestUtil.parseKey, TrainerTestUtil.parseBaseData, sc)
    val evalExamples = TestUtil.parseCSVToRDD(
      eval, TrainerTestUtil.parseKey, TrainerTestUtil.parseBaseData, sc)

    val r = trainer.
      searchBestOptionsPerModel(
        trainingExamples, evalExamples, evalExamples, Array(options)).collect()

    logger.info(s"size ${r.length}")
    logger.info(s"c ${r.head.toString}")
  }
}
