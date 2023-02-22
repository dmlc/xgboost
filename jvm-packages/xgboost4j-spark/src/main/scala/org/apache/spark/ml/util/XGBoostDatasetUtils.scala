package org.apache.spark.ml.util

import org.apache.spark.sql.Dataset

object XGBoostDatasetUtils {
  def getNumClasses(dataset: Dataset[_], labelCol: String, maxNumClasses: Int = 100): Int = {
    DatasetUtils.getNumClasses(dataset, labelCol, maxNumClasses)
  }
}
