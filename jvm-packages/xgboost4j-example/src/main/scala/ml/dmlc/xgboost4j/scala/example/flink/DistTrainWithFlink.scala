/*
 Copyright (c) 2014 - 2023 by Contributors

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
package ml.dmlc.xgboost4j.scala.example.flink

import java.lang.{Double => JDouble, Long => JLong}
import java.nio.file.{Path, Paths}
import org.apache.flink.api.java.tuple.{Tuple13, Tuple2}
import org.apache.flink.api.java.{DataSet, ExecutionEnvironment}
import org.apache.flink.ml.linalg.{Vector, Vectors}
import ml.dmlc.xgboost4j.java.flink.{XGBoost, XGBoostModel}
import org.apache.flink.api.common.typeinfo.{TypeHint, TypeInformation}
import org.apache.flink.api.java.utils.DataSetUtils


object DistTrainWithFlink {
  import scala.jdk.CollectionConverters._
  private val rowTypeHint = TypeInformation.of(new TypeHint[Tuple2[Vector, JDouble]]{})
  private val testDataTypeHint = TypeInformation.of(classOf[Vector])

  private[flink] def parseCsv(trainPath: Path)(implicit env: ExecutionEnvironment):
      DataSet[Tuple2[JLong, Tuple2[Vector, JDouble]]] = {
    DataSetUtils.zipWithIndex(
    env
      .readCsvFile(trainPath.toString)
      .ignoreFirstLine
      .types(
        classOf[Double], classOf[String], classOf[Double], classOf[Double], classOf[Double],
        classOf[Integer], classOf[Integer], classOf[Integer], classOf[Integer],
        classOf[Integer], classOf[Integer], classOf[Integer], classOf[Integer]
      )
      .map((row: Tuple13[Double, String, Double, Double, Double,
        Integer, Integer, Integer, Integer, Integer, Integer, Integer, Integer]) => {
        val dense = Vectors.dense(row.f2, row.f3, row.f4,
          row.f5.toDouble, row.f6.toDouble, row.f7.toDouble, row.f8.toDouble,
          row.f9.toDouble, row.f10.toDouble, row.f11.toDouble, row.f12.toDouble)
        val label = if (row.f1.contains("inf")) {
          JDouble.valueOf(1.0)
        } else {
          JDouble.valueOf(0.0)
        }
        new Tuple2[Vector, JDouble](dense, label)
      })
      .returns(rowTypeHint)
    )
  }

  private[flink] def runPrediction(trainPath: Path, percentage: Int)
                                  (implicit env: ExecutionEnvironment):
    (XGBoostModel, DataSet[Array[Float]]) = {
    // read training data
    val data: DataSet[Tuple2[JLong, Tuple2[Vector, JDouble]]] = parseCsv(trainPath)
    val trainSize = Math.round(0.01 * percentage * data.count())
    val trainData: DataSet[Tuple2[Vector, JDouble]] =
      data.filter(d => d.f0 < trainSize).map(_.f1).returns(rowTypeHint)


    val testData: DataSet[Vector] =
        data
          .filter(d => d.f0 >= trainSize)
          .map(_.f1.f0)
          .returns(testDataTypeHint)

    val paramMap = Map(
        ("eta", "0.1".asInstanceOf[AnyRef]),
        ("max_depth", "2"),
        ("objective", "binary:logistic"),
        ("verbosity", "1")
      )
      .asJava

    // number of iterations
    val round = 2
    // train the model
    val model = XGBoost.train(trainData, paramMap, round)
    val result = model.predict(testData).map(prediction => prediction.map(Float.unbox))
    (model, result)
  }

  def main(args: Array[String]): Unit = {
    implicit val env: ExecutionEnvironment = ExecutionEnvironment.getExecutionEnvironment
    val parentPath = Paths.get(args.headOption.getOrElse("."))
    val (_, predTest) = runPrediction(parentPath.resolve("veterans_lung_cancer.csv"), 70)
    val list = predTest.collect().asScala
    println(list.length)
  }
}
