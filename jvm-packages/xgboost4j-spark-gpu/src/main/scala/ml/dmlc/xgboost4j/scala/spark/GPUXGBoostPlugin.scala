package ml.dmlc.xgboost4j.scala.spark

import org.apache.spark.sql.Dataset

class GPUXGBoostPlugin extends XGBoostPlugin {

  /**
   * Whether the plugin is enabled or not, if not enabled, fallback
   * to the regular CPU pipeline
   *
   * @param dataset the input dataset
   * @return Boolean
   */
  override def isEnabled(dataset: Option[Dataset[_]]): Boolean = {
    false
  }
}
