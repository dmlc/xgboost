package com.airbnb.common.ml.xgboost.data

import ml.dmlc.xgboost4j.LabeledPoint
import org.apache.spark.sql.Row


trait TrainingModelData extends Serializable {

  def parseRowToXgboostLabeledPoint(row: Row): LabeledPoint = {
    // must preserve the order in select: label, node_id, features ...
    ModelData.parseRowToRawXgboostLabeledPoint(row, 0, 2)
  }
}

// default implementation for TrainingModelData
object BaseTrainingModelData extends TrainingModelData
