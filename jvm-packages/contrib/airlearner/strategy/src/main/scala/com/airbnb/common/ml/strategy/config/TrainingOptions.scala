package com.airbnb.common.ml.strategy.config

import com.typesafe.config.Config
import scala.collection.JavaConverters._
import scala.util.Try

import com.airbnb.common.ml.util.ConfigUtils


/**
  * A configuration options class that stores all potential options needed by trainers.
  */
case class TrainingOptions(
    trueLowerBound: Double,
    falseUpperBound: Double,
    falseLowerBound: Double,
    trueUpperBound: Double,
    r0: Double,
    r1: Double,
    rateDecay: Double,
    numEpochs: Int,
    miniBatchSize: Int,
    evalRatio: Double,
    minTrueLabelCount: Int,
    maxAvgLossRatio: Double,
    min: List[Double],
    max: List[Double],
    default: List[Double],
    dsEval: String,
    learningRateType: String,
    // This is an additional map, potentially provides a collection
    // of extra configurations for the trainer
    additionalOptions: Map[String, Any] = Map()
) {

  /**
    * Provides additional option given a key
    *
    * @param key
    * @param default
    * @tparam T expected value type
    * @return an option of the value
    */
  def getAdditionalOption[T](key: String, default: Option[T] = None): Option[T] = {
    if (additionalOptions.contains(key)) {
      Some(additionalOptions(key).asInstanceOf[T])
    } else {
      default
    }
  }

  // turn training Options to array of Double so that it can be saved
  def toArray: Array[Double] = {
    val array: Array[Double] = Array(trueLowerBound, falseUpperBound, falseLowerBound,
      trueUpperBound,
      r0, r1, rateDecay, numEpochs.toDouble, miniBatchSize.toDouble,
      evalRatio, minTrueLabelCount, maxAvgLossRatio) ++ min ++ max ++ default
    array
  }

  // only save limited fields, used by fromArrayAndGeneralOptions
  def toPartialArray: Array[Double] = {
    val array: Array[Double] = Array(
      trueLowerBound, falseUpperBound, falseLowerBound,
      trueUpperBound, evalRatio)
    array
  }

  def updateTrainingOptionsWithList(
      paramNames: List[String],
      paramValues: List[List[Double]]
  ): TrainingOptions = {
    TrainingOptions(
      trueLowerBound,
      falseUpperBound,
      falseLowerBound,
      trueUpperBound,
      r0,
      r1,
      rateDecay,
      numEpochs,
      miniBatchSize,
      evalRatio,
      minTrueLabelCount,
      maxAvgLossRatio,
      TrainingOptions.defaultOrNewValue(
        min, paramNames, paramValues, "min"),
      TrainingOptions.defaultOrNewValue(
        max, paramNames, paramValues, "max"),
      TrainingOptions.defaultOrNewValue(
        default, paramNames, paramValues, "default"),
      dsEval,
      learningRateType,
      additionalOptions)
  }

  def updateTrainingOptions(paramNames: List[String], paramValues: List[Double]): TrainingOptions
  = {
    TrainingOptions(
      // TODO replace k1/k2/lowerBound/upperBound
      TrainingOptions.defaultOrNewValue(
        trueLowerBound, paramNames, paramValues, "k1"),
      TrainingOptions.defaultOrNewValue(
        falseUpperBound, paramNames, paramValues, "k2"),
      TrainingOptions.defaultOrNewValue(
        falseLowerBound, paramNames, paramValues, "lowerBound"),
      TrainingOptions.defaultOrNewValue(
        trueUpperBound, paramNames, paramValues, "upperBound"),
      TrainingOptions.defaultOrNewValue(
        r0, paramNames, paramValues, "r0"),
      TrainingOptions.defaultOrNewValue(
        r1, paramNames, paramValues, "r1"),
      TrainingOptions.defaultOrNewValue(
        rateDecay, paramNames, paramValues, "rateDecay"),
      TrainingOptions.defaultOrNewValue(
        numEpochs.toDouble, paramNames, paramValues, "numEpochs").toInt,
      TrainingOptions.defaultOrNewValue(
        miniBatchSize.toDouble, paramNames, paramValues, "miniBatchSize").toInt,
      TrainingOptions.defaultOrNewValue(
        evalRatio, paramNames, paramValues, "eval_ratio"),
      TrainingOptions.defaultOrNewValue(
        minTrueLabelCount.toDouble, paramNames, paramValues, "min_true_label_count").toInt,
      TrainingOptions.defaultOrNewValue(
        maxAvgLossRatio, paramNames, paramValues, "max_avg_loss_ratio"),
      min,
      max,
      default,
      dsEval,
      learningRateType,
      updateAdditionalTrainingOptions(additionalOptions, paramNames, paramValues)
    )
  }

  private def updateAdditionalTrainingOptions(
      additionalOptions: Map[String, Any],
      paramNames: List[String],
      paramValues: List[Double]
  ): Map[String, Any] = {
    val replacement = paramNames.zipWithIndex.map { case (name, i) => {
      (name, paramValues(i))
    }
    }.toMap

    additionalOptions.map { case (key, value) =>
      // Override the original parameter if provided otherwise, retain the original one
    {
      (key, replacement.getOrElse(key, value))
    }
    }
  }
}

object TrainingOptions {

  def singleOptionsLength = 12

  def listOptionsLength = 3

  def defaultOrNewValue[T](
      default: T,
      paramNames: List[String],
      paramValues: List[T],
      name: String
  ): T = {
    val index = paramNames.indexOf(name)
    if (index >= 0) {
      paramValues(index)
    } else {
      default
    }
  }

  def getParamLength(totalLength: Int): Int = {
    (totalLength - singleOptionsLength) / listOptionsLength
  }

  private def getSlice(data: Array[Double], start: Int, size: Int): (List[Double], Int) = {
    val end = start + size
    (data.slice(start, end).toList, end)
  }

  // used to import from array generated by toPartialArray
  def fromArrayAndGeneralOptions(data: Array[Double], options: TrainingOptions): TrainingOptions = {
    TrainingOptions(
      data(0),
      data(1),
      data(2),
      data(3),
      options.r0,
      options.r1,
      options.rateDecay,
      options.numEpochs,
      options.miniBatchSize,
      data(4),
      options.minTrueLabelCount,
      options.maxAvgLossRatio,
      options.min,
      options.max,
      options.default,
      options.dsEval,
      options.learningRateType
    )
  }

  def fromArray(
      data: Array[Double],
      dsEval: String,
      learningRateType: String
  ): TrainingOptions = {
    val paramLength = getParamLength(data.length)

    val (min, maxStart) = getSlice(data, singleOptionsLength, paramLength)
    val (max, defaultStart) = getSlice(data, maxStart, paramLength)
    val (default, start) = getSlice(data, defaultStart, paramLength)

    TrainingOptions(
      data(0),
      data(1),
      data(2),
      data(3),
      data(4),
      data(5),
      data(6),
      data(7).toInt,
      data(8).toInt,
      data(9),
      data(10).toInt,
      data(11),
      min,
      max,
      default,
      dsEval,
      learningRateType
    )
  }

  def loadBaseTrainingOptions(config: Config): TrainingOptions = {
    // TODO update k1 k2 config name
    val trueLowerBound = Try(config.getDouble("k1")).getOrElse(1.0)
    val falseUpperBound = Try(config.getDouble("k2")).getOrElse(1.0)
    val falseLowerBound = Try(config.getDouble("lowerBound")).getOrElse(0.6)
    val trueUpperBound = Try(config.getDouble("upperBound")).getOrElse(3.0)
    val rateDecay = Try(config.getDouble("rateDecay")).getOrElse(1.0)
    val r0 = config.getDouble("r0")
    val r1 = config.getDouble("r1")
    val numIterations = Try(config.getInt("numEpochs")).getOrElse(50)
    val miniBatchSize = Try(config.getInt("miniBatchSize")).getOrElse(100)
    val evalRatio: Double = Try(config.getDouble("eval_ratio")).getOrElse(0.5)
    val minTrueLabelCount = Try(config.getInt("min_true_label_count")).getOrElse(0)
    val maxAvgLossRatio: Double = Try(config.getDouble("max_avg_loss_ratio")).getOrElse(0.0)

    val dsEval = config.getString("ds_eval")
    val min = config.getDoubleList("min").asScala.map(_.doubleValue()).toList
    val max = config.getDoubleList("max").asScala.map(_.doubleValue()).toList
    val default = Try(config.getDoubleList("default").asScala.map(_.doubleValue()).toList).
      getOrElse(List())

    val learningRateType = Try(config.getString("learning_rate_type")).getOrElse("default")

    val additionalConfig = Try(Some(
      config.getConfig("additional_options"))).getOrElse[Option[Config]](None)
    val additionalOptions: Map[String, Any] = additionalConfig match {
      case Some(config) => ConfigUtils.configToMap(config)
      case _ => Map()
    }

    TrainingOptions(
      trueLowerBound,
      falseUpperBound,
      falseLowerBound,
      trueUpperBound,
      r0,
      r1,
      rateDecay,
      numIterations,
      miniBatchSize,
      evalRatio,
      minTrueLabelCount,
      maxAvgLossRatio,
      min,
      max,
      default,
      dsEval,
      learningRateType,
      additionalOptions
    )
  }
}

