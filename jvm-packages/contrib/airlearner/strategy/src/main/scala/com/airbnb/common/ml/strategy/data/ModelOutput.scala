package com.airbnb.common.ml.strategy.data

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SaveMode}

import com.airbnb.common.ml.strategy.config.TrainingOptions
import com.airbnb.common.ml.strategy.eval.BinaryMetrics
import com.airbnb.common.ml.strategy.params.StrategyParams
import com.airbnb.common.ml.util.HiveManageTable


/**
  * ModelOutput represents the final result of a training and evaluation run.
  *
  * @param id modelId
  * @param params trained strategy params for this model
  * @param loss final loss score for this model
  * @param evalMetrics evaluation metrics (not currently saved)
  * @param holdoutMetrics holdout metrics (saved)
  * @param options training options used
  */
case class ModelOutput[T](
    id: String,
    params: StrategyParams[T],
    loss: Double,
    evalMetrics: BinaryMetrics,
    holdoutMetrics: BinaryMetrics,
    options: TrainingOptions
) extends HiveManageTable {
  override def toRow(partition: String): Row = {
    Row(
      id.toLong,
      holdoutMetrics.posCount,
      holdoutMetrics.negCount,
      holdoutMetrics.posSugHigher,
      holdoutMetrics.posSugLower,
      holdoutMetrics.negSugHigher,
      holdoutMetrics.negSugLower,
      holdoutMetrics.increasePrecision,
      holdoutMetrics.increaseRecall,
      holdoutMetrics.decreasePrecision,
      holdoutMetrics.decreaseRecall,
      holdoutMetrics.trueRegret,
      holdoutMetrics.trueRegretMedian,
      holdoutMetrics.trueRegret75Percentile,
      holdoutMetrics.falseRegret,
      holdoutMetrics.trueIncreaseMagnitude,
      holdoutMetrics.trueDecreaseMagnitude,
      holdoutMetrics.falseDecreaseMagnitude,
      holdoutMetrics.falseIncreaseMagnitude,
      params.params,
      loss,
      options.toPartialArray,
      partition
    )
  }
}

object ModelOutput {
  lazy val schema = StructType(
    Seq(
      StructField("id", LongType),
      StructField("posCount", IntegerType),
      StructField("negCount", IntegerType),
      StructField("posSugHigher", IntegerType),
      StructField("posSugLower", IntegerType),
      StructField("negSugHigher", IntegerType),
      StructField("negSugLower", IntegerType),

      StructField("increasePrecision", DoubleType),
      StructField("increaseRecall", DoubleType),
      StructField("decreasePrecision", DoubleType),
      StructField("decreaseRecall", DoubleType),

      StructField("trueRegret", DoubleType),
      StructField("trueRegretMedian", DoubleType),
      StructField("trueRegret75Percentile", DoubleType),

      StructField("falseRegret", DoubleType),
      StructField("trueIncreaseMagnitude", DoubleType),
      StructField("trueDecreaseMagnitude", DoubleType),
      StructField("falseDecreaseMagnitude", DoubleType),
      StructField("falseIncreaseMagnitude", DoubleType),

      StructField("params", ArrayType(DoubleType)),
      StructField("loss", DoubleType),
      StructField("options", ArrayType(DoubleType)),

      StructField("model", StringType)
    )
  )

  def save[T](
      hiveContext: HiveContext,
      data: RDD[ModelOutput[T]],
      table: String,
      partition: String
  ): Unit = {
    HiveManageTable.saveRDDToHive(
      hiveContext,
      data,
      table,
      ModelOutput.schema,
      SaveMode.Overwrite,
      "model",
      partition)
  }
}
