package ml.dmlc.xgboost4j.scala.spark

import org.apache.spark.sql.Dataset

trait XGBoostPlugin {
  /**
   * Whether the plugin is enabled or not, if not enabled, fallback
   * to the regular CPU pipeline
   * @param dataset the input dataset
   * @return Boolean
   */
  def isEnabled(dataset: Option[Dataset[_]]): Boolean


}
