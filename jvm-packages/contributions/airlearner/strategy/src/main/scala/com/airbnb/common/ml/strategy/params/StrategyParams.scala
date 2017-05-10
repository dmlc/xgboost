package com.airbnb.common.ml.strategy.params

import org.apache.spark.sql.Row

import com.airbnb.common.ml.strategy.config.TrainingOptions

trait StrategyParams [T] extends Serializable {
  def apply(update: Array[Double]): StrategyParams[T]

  def params: Array[Double]

  def getDefaultParams(trainingOptions: TrainingOptions): StrategyParams[T]

  // default all use field params in hive
  def parseParamsFromHiveRow(row: Row): StrategyParams [T] = {
    this(row.getAs[scala.collection.mutable.WrappedArray[Double]]("params").toArray)
  }

  def score(example: T): Double

  def updatedParams(grad: Array[Double], option: TrainingOptions): StrategyParams[T] = {
    // Project parameters to boxed constraints, very essential to limit parameter feasibly region
    val update = params.zip(grad).map(x=>x._1-x._2).
      zip(option.min).map(x => math.max(x._1, x._2)).
      zip(option.max).map(x => math.min(x._1, x._2))

    // project to additional constraints
    projectToAdditionalConstraints(update, option)

    this(update)
  }

  def projectToAdditionalConstraints(param: Array[Double], option: TrainingOptions): Unit = {
    // note: the constrained region has to be a convex hull
    // for now, we are just project it to a linear constraint:
    // b >= slope * a + intercept

    val slope: Double = option.additionalOptions.getOrElse("b_slope", 0d)
      .asInstanceOf[Number].doubleValue()
    val intercept: Double = option.additionalOptions.getOrElse("b_intercept", 0d)
      .asInstanceOf[Number].doubleValue()


    val bLowerBound = slope * param(0) + intercept
    // Project if the bound is larger than zero
    if (bLowerBound > 0d) {
      // also 0.5 <= b <= 1
      val u = Math.max(Math.min(1d, bLowerBound), 0.5d)
      param(1) = Math.max(param(1), u)
    }
  }

  // computeGradient computes all gradients
  def computeGradient(grad: Double,
                      example: T): Array[Double]

  def hasValidValue: Boolean = {
    valid(params)
  }

  def prettyPrint: String = {
    toString
  }

  // for save into ARRAY<DOUBLE> defined in strategy_model_dev_output
  // override it if not using same format or table
  override def toString: String = {
    params.mkString(StrategyParams.sepString)
  }

  /**
    * Check if all of the params in our param array are real numbers
    *
    * @param values params to check
    * @return true if none are invalid
    */
  private def valid(values: Array[Double]): Boolean = {
    for(value <- values) {
      if (value.isNaN) {
        return false
      }
    }
    true
  }

  /*
    Used in parameter search
    pass in \t separated string
    format: id_key parameter_search_index parameters
    output ((id, parameter_search_index), StrategyParams)
    default
  */
  def parseLine(line: String): ((java.lang.String, Int), StrategyParams[T]) = {
    val items: Array[String] = line.split("\t")
    assert(items.length == 3, s"line: $line ${items.mkString(",")}")
    val id = items(0)
    val paramIdx = items(1).toInt

    ((id, paramIdx), this(items(2).split(StrategyParams.sepString).map(_.toDouble)))
  }
}

object StrategyParams {
  val sepString = "\001"
}



