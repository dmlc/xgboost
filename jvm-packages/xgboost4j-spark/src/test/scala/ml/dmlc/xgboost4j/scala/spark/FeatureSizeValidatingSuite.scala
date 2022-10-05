/*
 Copyright (c) 2014-2022 by Contributors

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

import org.apache.spark.Partitioner
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.functions._

import scala.util.Random

class FeatureSizeValidatingSuite extends AnyFunSuite with PerTest {

  test("transform throwing exception if feature size of dataset is greater than model's") {
    val modelPath = getClass.getResource("/model/0.82/model").getPath
    val model = XGBoostClassificationModel.read.load(modelPath)
    val r = new Random(0)
    // 0.82/model was trained with 251 features. and transform will throw exception
    // if feature size of data is not equal to 251
    var df = ss.createDataFrame(Seq.fill(100)(r.nextInt(2)).map(i => (i, i))).
      toDF("feature", "label")
    for (x <- 1 to 252) {
      df = df.withColumn(s"feature_${x}", lit(1))
    }
    val assembler = new VectorAssembler()
      .setInputCols(df.columns.filter(!_.contains("label")))
      .setOutputCol("features")
    val thrown = intercept[Exception] {
      model.transform(assembler.transform(df)).show()
    }
    assert(thrown.getMessage.contains(
      "Number of columns does not match number of features in booster"))
  }

  test("train throwing exception if feature size of dataset is different on distributed train") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic",
      "num_round" -> 5, "num_workers" -> 2, "use_external_memory" -> true, "missing" -> 0)
    import ml.dmlc.xgboost4j.scala.spark.util.DataUtils._
    val sparkSession = ss
    import sparkSession.implicits._
    val repartitioned = sc.parallelize(Synthetic.trainWithDiffFeatureSize, 2)
      .map(lp => (lp.label, lp)).partitionBy(
      new Partitioner {
        override def numPartitions: Int = 2

        override def getPartition(key: Any): Int = key.asInstanceOf[Float].toInt
      }
    ).map(_._2).zipWithIndex().map {
      case (lp, id) =>
        (id, lp.label, lp.features)
    }.toDF("id", "label", "features")
    val xgb = new XGBoostClassifier(paramMap)
    xgb.fit(repartitioned)
  }
}
