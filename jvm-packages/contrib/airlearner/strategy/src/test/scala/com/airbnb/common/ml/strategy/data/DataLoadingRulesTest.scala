package com.airbnb.common.ml.strategy.data

import scala.util.Random

import org.junit.Assert
import org.junit.Test


class DataLoadingRulesTest {

  @Test
  def isEnoughSamplesToTrain(): Unit = {
    Assert.assertFalse(
      "There must be enough samples to allow training.",
      DataLoadingRules.isEnoughSamplesToTrain(
        Seq.fill(DataLoadingRules.MinTrainingSamples - 1)(Random.nextInt)
      )
    )

    Assert.assertTrue(
      "If there are enough samples, we should allow training.",
      DataLoadingRules.isEnoughSamplesToTrain(
        Seq.fill(DataLoadingRules.MinTrainingSamples)(Random.nextInt)
      )
    )
  }
}
