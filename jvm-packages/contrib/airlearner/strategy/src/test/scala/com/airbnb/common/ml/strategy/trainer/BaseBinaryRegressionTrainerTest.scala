package com.airbnb.common.ml.strategy.trainer

import org.junit.Test

import com.airbnb.common.ml.strategy.data.BaseBinarySample
import com.airbnb.common.ml.strategy.params.BaseParam
import com.airbnb.common.ml.test.TestWithHiveContext
import com.airbnb.common.ml.util.{ScalaLogging, TestUtil}


class BaseBinaryRegressionTrainerTest
  extends TestWithHiveContext
    with ScalaLogging {

  @Test
  def searchAllBestOptions(): Unit = {
    val trainer = BaseBinaryRegressionTrainer(BaseParam(), BaseBinarySample)
    TrainerTestUtil.testSearchAllBestOptions("/data/train.csv", "/data/eval.csv", hc, sc, trainer)
  }

  @Test
  def getParamsFromPerModelOptions(): Unit = {
    val params = BaseParam.getDefault

    val trainer = BaseBinaryRegressionTrainer(params, BaseBinarySample)
    val trainingOptions = TrainerTestUtil.getOptionsTanhv2()
    val perModelOptions = sc.parallelize(
      Array((TrainerTestUtil.key, Array(
        0.98, 1.0,
        0.65, 1.03,
        1.0))))
    val trainingExamples = TestUtil.parseCSVToRDD(
      "/data/train.csv", TrainerTestUtil.parseKey, TrainerTestUtil.parseBaseData, sc)

    val result = trainer.getParamsFromPerModelOptions(
      trainingExamples,
      perModelOptions,
      trainingOptions
    ).collect()

    logger.info(s"getParamsFromPerModelOptions ${result.mkString(",")}")
  }

}
