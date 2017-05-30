package com.airbnb.common.ml.strategy.params

import com.airbnb.common.ml.strategy.config.TrainingOptions
import com.airbnb.common.ml.strategy.data.BaseBinarySample


// use tanh function and assume BinaryScoringSample.x is between 0~1
case class BaseParam (params: Array[Double] = Array()) extends StrategyParams[BaseBinarySample] {
  override def apply(update: Array[Double]): StrategyParams[BaseBinarySample] = {
    BaseParam(update)
  }

  override def getDefaultParams(trainingOptions: TrainingOptions): StrategyParams[BaseBinarySample] = {
    if (trainingOptions.default.length == 3) {
      BaseParam(trainingOptions.default.toArray)
    } else {
      BaseParam.getDefault
    }
  }

  override def score(example: BaseBinarySample): Double = {
    (1 + params(0) * math.tanh(probOffset(example))) * example.scoringPivot
  }

  private def probOffset(example: BaseBinarySample): Double = {
    example.x * params(1) + params(2)
  }

  override def computeGradient(grad: Double, example: BaseBinarySample): Array[Double] = {
    val prob = probOffset(example)
    val gradA = grad * math.tanh(prob)
    // sech(x) = 1/ cosh(x)
    val cosh = math.cosh(prob)
    val gradC = grad * params(0) / (cosh * cosh)
    val gradB = gradC * params(1)
    Array(gradA, gradB, gradC)
  }
}

object BaseParam {
  def getDefault: BaseParam = {
    BaseParam(Array(0.2, 20, -12.0))
  }
}
