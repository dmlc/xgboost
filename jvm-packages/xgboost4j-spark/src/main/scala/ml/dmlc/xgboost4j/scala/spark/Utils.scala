/*
 Copyright (c) 2014-2025 by Contributors

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

import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.json4s.{DefaultFormats, FullTypeHints, JField, JValue, NoTypeHints, TypeHints}

import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}

private[scala] object Utils {

  private[spark] implicit class XGBLabeledPointFeatures(
      val labeledPoint: XGBLabeledPoint
  ) extends AnyVal {
    /** Converts the point to [[MLLabeledPoint]]. */
    private[spark] def asML: MLLabeledPoint = {
      MLLabeledPoint(labeledPoint.label, labeledPoint.features)
    }

    /**
     * Returns feature of the point as [[org.apache.spark.ml.linalg.Vector]].
     */
    def features: Vector = if (labeledPoint.indices == null) {
      Vectors.dense(labeledPoint.values.map(_.toDouble))
    } else {
      Vectors.sparse(labeledPoint.size, labeledPoint.indices, labeledPoint.values.map(_.toDouble))
    }
  }

  private[spark] implicit class MLVectorToXGBLabeledPoint(val v: Vector) extends AnyVal {
    /**
     * Converts a [[Vector]] to a data point with a dummy label.
     *
     * This is needed for constructing a [[ml.dmlc.xgboost4j.scala.DMatrix]]
     * for prediction.
     */
    // TODO support sparsevector
    def asXGB: XGBLabeledPoint = v match {
      case v: DenseVector =>
        XGBLabeledPoint(0.0f, v.size, null, v.values.map(_.toFloat))
      case v: SparseVector =>
        XGBLabeledPoint(0.0f, v.size, v.indices, v.toDense.values.map(_.toFloat))
    }
  }

  def getSparkClassLoader: ClassLoader = getClass.getClassLoader

  def getContextOrSparkClassLoader: ClassLoader =
    Option(Thread.currentThread().getContextClassLoader).getOrElse(getSparkClassLoader)

  // scalastyle:off classforname

  /** Preferred alternative to Class.forName(className) */
  def classForName(className: String): Class[_] = {
    Class.forName(className, true, getContextOrSparkClassLoader)
    // scalastyle:on classforname
  }

  /**
   * Get the TypeHints according to the value
   *
   * @param value the instance of class to be serialized
   * @return if value is null,
   *         return NoTypeHints
   *         else return the FullTypeHints.
   *
   *         The FullTypeHints will save the full class name into the "jsonClass" of the json,
   *         so we can find the jsonClass and turn it to FullTypeHints when deserializing.
   */
  def getTypeHintsFromClass(value: Any): TypeHints = {
    if (value == null) { // XGBoost will save the default value (null)
      NoTypeHints
    } else {
      FullTypeHints(List(value.getClass))
    }
  }

  /**
   * Get the TypeHints according to the saved jsonClass field
   *
   * @param json
   * @return TypeHints
   */
  def getTypeHintsFromJsonClass(json: JValue): TypeHints = {
    val jsonClassField = json findField {
      case JField("jsonClass", _) => true
      case _ => false
    }

    jsonClassField.map { field =>
      implicit val formats = DefaultFormats
      val className = field._2.extract[String]
      FullTypeHints(List(Utils.classForName(className)))
    }.getOrElse(NoTypeHints)
  }

  val TRAIN_NAME = "train"
  val VALIDATION_NAME = "eval"


  /** Executes the provided code block and then closes the resource */
  def withResource[T <: AutoCloseable, V](r: T)(block: T => V): V = {
    try {
      block(r)
    } finally {
      r.close()
    }
  }

  /** Executes the provided code block and then closes the sequence of resources */
  def withResource[T <: AutoCloseable, V](r: Seq[T])(block: Seq[T] => V): V = {
    try {
      block(r)
    } finally {
      r.safeClose()
    }
  }

  implicit class AutoCloseableSeq[A <: AutoCloseable](val in: collection.SeqLike[A, _]) {
    /**
     * safeClose: Is an implicit on a sequence of AutoCloseable classes that tries to close each
     * element of the sequence, even if prior close calls fail. In case of failure in any of the
     * close calls, an Exception is thrown containing the suppressed exceptions (getSuppressed),
     * if any.
     */
    def safeClose(error: Throwable = null): Unit = if (in != null) {
      var closeException: Throwable = null
      in.foreach { element =>
        if (element != null) {
          try {
            element.close()
          } catch {
            case e: Throwable if error != null => error.addSuppressed(e)
            case e: Throwable if closeException == null => closeException = e
            case e: Throwable => closeException.addSuppressed(e)
          }
        }
      }
      if (closeException != null) {
        // an exception happened while we were trying to safely close
        // resources, throw the exception to alert the caller
        throw closeException
      }
    }
  }
}
