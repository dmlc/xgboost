/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.spark

import scala.collection.mutable

import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import org.apache.spark.ml.linalg.{Vector => MLVector, DenseVector => MLDenseVector}
import org.apache.spark.ml.param.{DoubleArrayParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class XGBoostClassificationModel private[spark](
    override val uid: String, _booster: Booster)
    extends XGBoostModel(_booster) {

  def this(_booster: Booster) = this(Identifiable.randomUID("XGBoostClassificationModel"), _booster)

  // scalastyle:off

  final val outputMargin: Param[Boolean] = new Param[Boolean](this, "outputMargin", "whether to output untransformed margin value ")

  setDefault(outputMargin, false)

  def setOutputMargin(value: Boolean): XGBoostModel = set(outputMargin, value).asInstanceOf[XGBoostClassificationModel]

  final val rawPredictionCol: Param[String] = new Param[String](this, "rawPredictionCol", "Column name for raw prediction output of xgboost. If outputMargin is true, the column contains untransformed margin value; otherwise it is the probability for each class (by default).")

  setDefault(rawPredictionCol, "probabilities")

  final def getRawPredictionCol: String = $(rawPredictionCol)

  def setRawPredictionCol(value: String): XGBoostClassificationModel = set(rawPredictionCol, value).asInstanceOf[XGBoostClassificationModel]

  final val thresholds: DoubleArrayParam = new DoubleArrayParam(this, "thresholds", "Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values >= 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class' threshold", (t: Array[Double]) => t.forall(_ >= 0))

  def getThresholds: Array[Double] = $(thresholds)

  def setThresholds(value: Array[Double]): XGBoostClassificationModel =
    set(thresholds, value).asInstanceOf[XGBoostClassificationModel]

  // scalastyle:on

  private def predictRaw(
      testSet: Dataset[_],
      temporalColName: Option[String] = None,
      forceTransformedScore: Option[Boolean] = None): DataFrame = {
    val predictRDD = produceRowRDD(testSet, forceTransformedScore.getOrElse($(outputMargin)))
    testSet.sparkSession.createDataFrame(predictRDD, schema = {
      StructType(testSet.schema.add(StructField(
        temporalColName.getOrElse($(rawPredictionCol)),
        ArrayType(FloatType, containsNull = false), nullable = false)))
    })
  }

  private def fromFeatureToPrediction(testSet: Dataset[_]): Dataset[_] = {
    val rawPredictionDF = predictRaw(testSet, Some("rawPredictionCol"))
    val predictionUDF = udf(raw2prediction _).apply(col("rawPredictionCol"))
    val tempDF = rawPredictionDF.withColumn($(predictionCol), predictionUDF)
    val allColumnNames = testSet.columns ++ Seq($(predictionCol))
    tempDF.select(allColumnNames(0), allColumnNames.tail: _*)
  }

  private def argMax(vector: mutable.WrappedArray[Float]): Double = {
    vector.zipWithIndex.maxBy(_._1)._2
  }

  private def raw2prediction(rawPrediction: mutable.WrappedArray[Float]): Double = {
    if (!isDefined(thresholds)) {
      argMax(rawPrediction)
    } else {
      probability2prediction(rawPrediction)
    }
  }

  private def probability2prediction(probability: mutable.WrappedArray[Float]): Double = {
    if (!isDefined(thresholds)) {
      argMax(probability)
    } else {
      val thresholds: Array[Double] = getThresholds
      val scaledProbability: mutable.WrappedArray[Double] =
        probability.zip(thresholds).map { case (p, t) =>
          if (t == 0.0) Double.PositiveInfinity else p / t
        }
      argMax(scaledProbability.map(_.toFloat))
    }
  }

  override protected def transformImpl(testSet: Dataset[_]): DataFrame = {
    transformSchema(testSet.schema, logging = true)
    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".transform() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }
    if ($(outputMargin)) {
      setRawPredictionCol("margin")
    }
    var outputData = testSet
    var numColsOutput = 0
    if ($(rawPredictionCol).nonEmpty) {
      outputData = predictRaw(testSet)
      numColsOutput += 1
    }

    if ($(predictionCol).nonEmpty) {
      if ($(rawPredictionCol).nonEmpty) {
        require(!$(outputMargin), "XGBoost does not support output final prediction with" +
          " untransformed margin. Please set predictionCol as \"\" when setting outputMargin as" +
          " true")
        val rawToPredUDF = udf(raw2prediction _).apply(col($(rawPredictionCol)))
        outputData = outputData.withColumn($(predictionCol), rawToPredUDF)
      } else {
        outputData = fromFeatureToPrediction(testSet)
      }
      numColsOutput += 1
    }

    if (numColsOutput == 0) {
      this.logWarning(s"$uid: XGBoostClassificationModel.transform() was called as NOOP" +
        " since no output columns were set.")
    }
    outputData.toDF()
  }

  private[spark] var numOfClasses = 2

  def numClasses: Int = numOfClasses

  override def copy(extra: ParamMap): XGBoostClassificationModel = {
    defaultCopy(extra)
  }

  override protected def predict(features: MLVector): Double = {
    throw new Exception("XGBoost does not support online prediction ")
  }
}
