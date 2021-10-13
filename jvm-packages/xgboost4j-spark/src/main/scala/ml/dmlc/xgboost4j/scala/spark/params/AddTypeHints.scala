package ml.dmlc.xgboost4j.scala.spark.params

trait AddTypeHints {
    val typeHintAdded = SavedTypeHints.addClassOf(this)
}
