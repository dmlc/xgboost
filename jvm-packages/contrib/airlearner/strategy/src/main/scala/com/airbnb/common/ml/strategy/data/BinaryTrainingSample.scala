package com.airbnb.common.ml.strategy.data

import com.airbnb.common.ml.strategy.config.TrainingOptions


/*
  Binary regression sample has the property of both binary classification and regression.
  it has a boolean label just like binary classification and observedValue just like regression.
  If label is true, observedValue is gained
  If label is false, 0 is gained.
  If we only use true label, this is a traditional regression problem
  pivot is used as a base estimation for predication.
  there are scoringPivot and basePivot in BinaryScoringSample
  BinaryScoringSample.scoringPivot is used in scoring.
  BinaryScoringSample.basePivot is used in training.
  BinaryScoringSample allows two pivots due to the first implementation of BinaryScoringSample
  use two pivots.
  BinaryScoringSample.x is the feature associated with observedValue or pivotValue
  i.e. x = probability of observedValue is true.
 */
trait BinaryTrainingSample extends BinaryScoringSample {

  def label: Boolean

  // value in label = true case
  // in false case, value has no meaning.
  protected def trueValue: Option[Double]

  def getTrueValue: Double = {
    assert(label)
    trueValue.get
  }

  /**
    * Loss is the absolute amount that a true prediction is low or that
    * a false prediction is high, compared to the observed value.
    *
    * @param prediction predicted value
    * @return loss amount
    */
  def lossWithPrediction(prediction: Double, ratio: Double): Double = {
    if (label) {
      ratio * Math.max(0, getTrueValue - prediction)
    } else {
      (1-ratio) * Math.max(0, prediction - observedValue)
    }
  }

  /**
    * Relative loss ratio, compared to the observed value.
    *
    * @param prediction predicted value
    * @return relative loss ratio
    */
  def lossRatioWithPrediction(prediction: Double, ratio: Double): Double = {
    val absLoss = lossWithPrediction(prediction, ratio)
    val denominator = if (label) {
      getTrueValue
    } else {
      observedValue
    }
    absLoss / denominator
  }

  def getTrueOrPivotValue: Double = {
    if (label) {
      getTrueValue
    } else {
      basePivot
    }
  }

  def getMinValue: Double = {
    math.min(observedValue, basePivot)
  }

  def getLowerBound(options: TrainingOptions): Double = {
    if (label) {
      getLowerBoundByScale(
        options.trueLowerBound,
        getTrueOrPivotValue)
    } else {
      getLowerBoundByScale(
        options.falseLowerBound,
        getMinValue)
    }
  }

  def getLowerBoundByScale(bound: Double, bottomLine: Double): Double = {
    val len: Double = (1 - bound) * 2
    val newBound: Double = 1 - len * (1 - x)
    newBound * bottomLine
  }

  def getUpperBound(options: TrainingOptions): Double = {
    if (label) {
      getUpperBoundForTrueLabel(options)
    } else {
      options.falseUpperBound * getMinValue
    }
  }

  /**
    * For a true label, get the upper bound for training.
    * The lower the prob, the smaller the upper bound.
    *
    * if p = 0; upperBound == 1.0
    * if p = 1; upperBound > 1.0
    *
    * @param options training options
    * @return upper bound
    */
  def getUpperBoundForTrueLabel(options: TrainingOptions): Double = {
    val len: Double = (options.trueUpperBound - 1) * 2
    val bound: Double = 1 + len * x
    // todo test with trueValue
    bound * basePivot
  }
}
