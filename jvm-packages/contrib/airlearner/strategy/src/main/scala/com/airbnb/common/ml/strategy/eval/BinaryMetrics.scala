package com.airbnb.common.ml.strategy.eval

import scala.util.Try

import org.apache.spark.rdd.RDD


case class BinaryMetrics(
    posCount: Int,
    negCount: Int,
    posSugHigher: Int,
    posSugLower: Int,
    negSugHigher: Int,
    negSugLower: Int,
    increasePrecision: Double,
    increaseRecall: Double,
    decreasePrecision: Double,
    decreaseRecall: Double,
    trueRegret: Double,
    trueRegretMedian: Double,
    trueRegret75Percentile: Double,
    falseRegret: Double,
    trueIncreaseMagnitude: Double,
    trueDecreaseMagnitude: Double,
    falseDecreaseMagnitude: Double,
    falseIncreaseMagnitude: Double,
    trueDecreaseSum: Double,
    trueIncreaseSum: Double,
    falseDecreaseSum: Double,
    falseIncreaseSum: Double
) {
  def toTSVRow: String = {
    Vector(
      posCount, negCount, posSugHigher, posSugLower, negSugHigher, negSugLower, // raw counts
      // precision-recall
      increasePrecision, increaseRecall, decreasePrecision, decreaseRecall,
      trueRegret, trueRegretMedian, trueRegret75Percentile, falseRegret,
      trueIncreaseMagnitude,
      trueDecreaseMagnitude,
      falseDecreaseMagnitude,
      falseIncreaseMagnitude // magnitude metrics
    ).mkString("\t")
  }

  override def toString: String = {
    Vector(
      // save to database
      toTSVRow,
      // 4 additional magnitude metrics
      trueDecreaseSum, trueIncreaseSum, falseDecreaseSum, falseIncreaseSum,
      // 2 loss metrics
      falseIncreaseSum / negSugHigher,
      trueDecreaseSum / posSugLower
    ).mkString("\t")
  }

  // For ease of printing with field names
  def toArray: Array[(String, Any)] = {
    this.getClass
      .getDeclaredFields
      .map(_.getName) // all field names
      .zip(this.productIterator.to)
  }

  def +(that: BinaryMetrics): BinaryMetrics = {
    BinaryMetrics(
      this.posCount + that.posCount,
      this.negCount + that.negCount,
      this.posSugHigher + that.posSugHigher,
      this.posSugLower + that.posSugLower,
      this.negSugHigher + that.negSugHigher,
      this.negSugLower + that.negSugLower,
      // metrics can't be added
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      this.trueDecreaseSum + that.trueDecreaseSum,
      this.trueIncreaseSum + that.trueIncreaseSum,
      this.falseDecreaseSum + that.falseDecreaseSum,
      this.falseIncreaseSum + that.falseIncreaseSum
    )
  }

  def recompute: BinaryMetrics = {
    val lowerCount = posSugLower + negSugLower
    val higherCount = posSugHigher + negSugHigher
    BinaryMetrics(
      posCount = posCount,
      negCount = negCount,
      posSugHigher = posSugHigher,
      posSugLower = posSugLower,
      negSugHigher = negSugHigher,
      negSugLower = negSugLower,
      increasePrecision = BinaryMetrics.safeDiv(posSugHigher, higherCount),
      increaseRecall = BinaryMetrics.safeDiv(posSugHigher, posCount),
      decreasePrecision = BinaryMetrics.safeDiv(negSugLower, lowerCount),
      decreaseRecall = BinaryMetrics.safeDiv(negSugLower, negCount),
      trueRegret = BinaryMetrics.safeDiv(trueDecreaseSum, posCount),
      trueRegretMedian = trueRegretMedian,
      trueRegret75Percentile = trueRegret75Percentile,
      falseRegret = BinaryMetrics.safeDiv(falseIncreaseSum, negCount),
      trueIncreaseMagnitude = BinaryMetrics.safeDiv(trueIncreaseSum, posSugHigher),
      trueDecreaseMagnitude = BinaryMetrics.safeDiv(trueDecreaseSum, posSugLower),
      falseDecreaseMagnitude = BinaryMetrics.safeDiv(falseDecreaseSum, negSugLower),
      falseIncreaseMagnitude = BinaryMetrics.safeDiv(falseIncreaseSum, negSugHigher),
      trueDecreaseSum,
      trueIncreaseSum,
      falseDecreaseSum,
      falseIncreaseSum
    )
  }
}

object BinaryMetrics {
  final val metricNames: Seq[String] =
    Vector(
      "posCount", "negCount", "posSugHigher", "posSugLower", "negSugHigher", "negSugLower",
      "increasePrecision", "increaseRecall", "decreasePrecision", "decreaseRecall",
      "trueRegret", "trueRegretMedian", "trueRegret75Percentile", "falseRegret",
      "trueIncreaseMagnitude",
      "trueDecreaseMagnitude",
      "falseDecreaseMagnitude",
      "falseIncreaseMagnitude"
    )

  final val metricsHeader: String = metricNames.mkString("\t")

  /**
    * Compute our evaluation metrics for a set of prediction results.
    *
    * @param results prediction results to eval: (label, predictionLower) -> (count, sum)
    * @param trueRegretMedian median true regret score
    * @param trueRegret75Percentile 75th percentil true regret
    * @return a populated BinaryMetrics instance
    */
  def computeEvalMetricFromCounts(
      results: Map[(Boolean, Boolean), (Int, Double)],
      trueRegretMedian: Double,
      trueRegret75Percentile: Double
  ): BinaryMetrics = {
    val posSugHigher: Int = Try(results((true, false))._1).getOrElse(0)
    val posSugLower: Int = Try(results((true, true))._1).getOrElse(0)
    val negSugHigher: Int = Try(results((false, false))._1).getOrElse(0)
    val negSugLower: Int = Try(results((false, true))._1).getOrElse(0)

    val posCount: Int = posSugHigher + posSugLower
    val negCount: Int = negSugHigher + negSugLower
    val lowerCount: Int = posSugLower + negSugLower
    val higherCount: Int = posSugHigher + negSugHigher

    val trueDecreaseSum: Double = Try(results((true, true))._2).getOrElse(0.0)
    val trueIncreaseSum: Double = Try(results((true, false))._2).getOrElse(0.0)
    val falseDecreaseSum: Double = Try(results((false, true))._2).getOrElse(0.0)
    val falseIncreaseSum: Double = Try(results((false, false))._2).getOrElse(0.0)

    BinaryMetrics(
      posCount = posCount,
      negCount = negCount,
      posSugHigher = posSugHigher,
      posSugLower = posSugLower,
      negSugHigher = negSugHigher,
      negSugLower = negSugLower,
      increasePrecision = safeDiv(posSugHigher, higherCount),
      increaseRecall = safeDiv(posSugHigher, posCount),
      decreasePrecision = safeDiv(negSugLower, lowerCount),
      decreaseRecall = safeDiv(negSugLower, negCount),
      trueRegret = safeDiv(trueDecreaseSum, posCount),
      trueRegretMedian = trueRegretMedian,
      trueRegret75Percentile = trueRegret75Percentile,
      falseRegret = safeDiv(falseIncreaseSum, negCount),
      trueIncreaseMagnitude = safeDiv(trueIncreaseSum, posSugHigher),
      trueDecreaseMagnitude = safeDiv(trueDecreaseSum, posSugLower),
      falseDecreaseMagnitude = safeDiv(falseDecreaseSum, negSugLower),
      falseIncreaseMagnitude = safeDiv(falseIncreaseSum, negSugHigher),
      trueDecreaseSum = trueDecreaseSum,
      trueIncreaseSum = trueIncreaseSum,
      falseDecreaseSum = falseDecreaseSum,
      falseIncreaseSum = falseIncreaseSum
    )
  }

  def combineEvalMetricFromRDD(data: RDD[BinaryMetrics]): BinaryMetrics = {
    val metrics = data.reduce((a, b) => {
      a + b
    })

    metrics.recompute
  }

  def safeDiv(numerator: Double, denominator: Double): Double = {
    if (denominator == 0) {
      0
    } else {
      numerator / denominator
    }
  }
}
