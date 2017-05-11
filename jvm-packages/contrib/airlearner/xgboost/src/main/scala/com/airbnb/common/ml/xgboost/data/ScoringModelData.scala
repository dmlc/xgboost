package com.airbnb.common.ml.xgboost.data

import org.apache.spark.sql.Row


trait ScoringModelData extends Serializable {
  def parseRowToXgboostLabeledPointAndData(row: Row): ScoringLabeledPoint
}
