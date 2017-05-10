package com.airbnb.common.ml.strategy.data

import com.airbnb.common.ml.strategy.eval.BinaryMetrics


trait BinaryScoringSample
  extends Serializable {

  // the main feature in the sample.
  // i.e. for pricing problem, this is the price seen by users
  def observedValue: Double

  // the value used as base for scoring
  // i.e. min(observed value) or observed value
  def scoringPivot: Double

  // a more stabled value than observedValue
  // can be avg(observedValue) or avg(trueValue) or other format
  // we allow scoringPivot != basePivot due to the very first implementation
  // use a different scoringPivot improved performance
  def basePivot: Double

  // the feature associated with observedValue or pivotValue
  // i.e. probability of the value
  def x: Double

  // TODO maybe need predictionHigher?
  def predictionLower(prediction: Double): Boolean = {
    prediction < observedValue
  }

  def predictionIncrease(prediction: Double): Double = {
    BinaryMetrics.safeDiv(prediction - observedValue, observedValue)
  }

  // return value between observedValue and basePivot scaled by x
  protected def scaleObservedValueByX: Double = {
    if (observedValue != basePivot) {
      val min = math.min(observedValue, basePivot)
      val max = math.max(observedValue, basePivot)
      min + (max - min) * x
    } else {
      basePivot
    }
  }

  protected def observedOrPivot: Double = {
    if (observedValue != basePivot) {
      math.min(observedValue, basePivot)
    } else {
      basePivot
    }
  }

}
