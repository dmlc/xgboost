package com.airbnb.common.ml.strategy.data

/**
  * Support data rules for loading input samples into the strategy executor pipeline.
  */
object DataLoadingRules {

  final val MinTrainingSamples: Int = 100

  /**
    * Is this sample set large enough to train the strategy model?
    *
    * @param samples samples to evaluate for this model
    * @return true if it is enough, otherwise false
    */
  def isEnoughSamplesToTrain[T](samples: Seq[T]): Boolean = {
    samples.length >= MinTrainingSamples
  }
}
