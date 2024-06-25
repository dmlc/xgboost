/*
 Copyright (c) 2024 by Contributors

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

import java.io.File

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{CSVOptions, DType, Schema, Table}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{FloatType, StructField, StructType}

import ml.dmlc.xgboost4j.scala.rapids.spark.GpuTestSuite

class GpuXGBoostPluginSuite extends GpuTestSuite {

  test("isEnabled") {
    def checkIsEnabled(spark: SparkSession, expected: Boolean): Unit = {
      import spark.implicits._
      val df = Seq((1.0f, 2.0f, 0.0f),
        (2.0f, 3.0f, 1.0f)
      ).toDF("c1", "c2", "label")
      val classifier = new XGBoostClassifier()
      assert(classifier.getPlugin.isDefined)
      assert(classifier.getPlugin.get.isEnabled(df) === expected)
    }

    withCpuSparkSession() { spark =>
      checkIsEnabled(spark, false)
    }

    withGpuSparkSession() { spark =>
      checkIsEnabled(spark, true)
    }
  }


  test("parameter validation") {
    withGpuSparkSession() { spark =>
      import spark.implicits._
      val df = Seq((1.0f, 2.0f, 1.0f, 2.0f, 0.0f, 0.0f),
        (2.0f, 3.0f, 2.0f, 3.0f, 1.0f, 0.1f),
        (3.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.1f),
        (4.0f, 5.0f, 6.0f, 7.0f, 0.0f, 0.1f),
        (5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.1f)
      ).toDF("c1", "c2", "weight", "margin", "label", "other")
      val classifier = new XGBoostClassifier()

      val plugin = classifier.getPlugin.get.asInstanceOf[GpuXGBoostPlugin]
      intercept[IllegalArgumentException] {
        plugin.validate(classifier, df)
      }
      classifier.setDevice("cuda")
      plugin.validate(classifier, df)

      classifier.setDevice("gpu")
      plugin.validate(classifier, df)

      classifier.setDevice("cpu")
      classifier.setTreeMethod("gpu_hist")
      plugin.validate(classifier, df)
    }
  }

  test("preprocess") {
    withGpuSparkSession() { spark =>
      import spark.implicits._
      val df = Seq((1.0f, 2.0f, 1.0f, 2.0f, 0.0f, 0.0f),
        (2.0f, 3.0f, 2.0f, 3.0f, 1.0f, 0.1f),
        (3.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.1f),
        (4.0f, 5.0f, 6.0f, 7.0f, 0.0f, 0.1f),
        (5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.1f)
      ).toDF("c1", "c2", "weight", "margin", "label", "other")
        .repartition(5)

      assert(df.schema.names.contains("other"))
      assert(df.rdd.getNumPartitions === 5)

      val features = Array("c1", "c2")
      var classifier = new XGBoostClassifier()
        .setNumWorkers(3)
        .setFeaturesCol(features)
      assert(classifier.getPlugin.isDefined)
      assert(classifier.getPlugin.get.isInstanceOf[GpuXGBoostPlugin])
      var out = classifier.getPlugin.get.asInstanceOf[GpuXGBoostPlugin]
        .preprocess(classifier, df)

      assert(out.schema.names.contains("c1") && out.schema.names.contains("c2"))
      assert(out.schema.names.contains(classifier.getLabelCol))
      assert(!out.schema.names.contains("weight") && !out.schema.names.contains("margin"))
      assert(out.rdd.getNumPartitions === 3)

      classifier = new XGBoostClassifier()
        .setNumWorkers(4)
        .setFeaturesCol(features)
        .setWeightCol("weight")
        .setBaseMarginCol("margin")
        .setDevice("cuda")
      out = classifier.getPlugin.get.asInstanceOf[GpuXGBoostPlugin]
        .preprocess(classifier, df)

      assert(out.schema.names.contains("c1") && out.schema.names.contains("c2"))
      assert(out.schema.names.contains(classifier.getLabelCol))
      assert(out.schema.names.contains("weight") && out.schema.names.contains("margin"))
      assert(out.rdd.getNumPartitions === 4)
    }
  }

  test("build RDD Watches") {
    withGpuSparkSession() { spark =>
      import spark.implicits._

      // dataPoint -> (missing, rowNum, nonMissing)
      Map(0.0f -> (0.0f, 5, 9), Float.NaN -> (0.0f, 5, 9)).foreach {
        case (data, (missing, expectedRowNum, expectedNonMissing)) =>
          val df = Seq(
            (1.0f, 2.0f, 1.0f, 2.0f, 0.0f, 0.0f),
            (2.0f, 3.0f, 2.0f, 3.0f, 1.0f, 0.1f),
            (3.0f, data, 5.0f, 6.0f, 0.0f, 0.1f),
            (4.0f, 5.0f, 6.0f, 7.0f, 0.0f, 0.1f),
            (5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 0.1f)
          ).toDF("c1", "c2", "weight", "margin", "label", "other")

          val features = Array("c1", "c2")
          val classifier = new XGBoostClassifier()
            .setNumWorkers(2)
            .setWeightCol("weight")
            .setBaseMarginCol("margin")
            .setFeaturesCol(features)
            .setDevice("cuda")
            .setMissing(missing)

          val rdd = classifier.getPlugin.get.buildRddWatches(classifier, df)
          val result = rdd.mapPartitions { iter =>
            val watches = iter.next()
            val size = watches.size
            val labels = watches.datasets(0).getLabel
            val weight = watches.datasets(0).getWeight
            val margins = watches.datasets(0).getBaseMargin
            val rowNumber = watches.datasets(0).rowNum
            val nonMissing = watches.datasets(0).nonMissingNum
            Iterator.single(size, rowNumber, nonMissing, labels, weight, margins)
          }.collect()

          val labels: ArrayBuffer[Float] = ArrayBuffer.empty
          val weight: ArrayBuffer[Float] = ArrayBuffer.empty
          val margins: ArrayBuffer[Float] = ArrayBuffer.empty
          val rowNumber: ArrayBuffer[Long] = ArrayBuffer.empty
          val nonMissing: ArrayBuffer[Long] = ArrayBuffer.empty

          for (row <- result) {
            assert(row._1 === 1)
            rowNumber.append(row._2)
            nonMissing.append(row._3)
            labels.append(row._4: _*)
            weight.append(row._5: _*)
            margins.append(row._6: _*)
          }
          assert(labels.sorted === Array(0.0f, 1.0f, 0.0f, 0.0f, 1.0f).sorted)
          assert(weight.sorted === Array(1.0f, 2.0f, 5.0f, 6.0f, 7.0f).sorted)
          assert(margins.sorted === Array(2.0f, 3.0f, 6.0f, 7.0f, 8.0f).sorted)
          assert(rowNumber.sum === expectedRowNum)
          assert(nonMissing.sum === expectedNonMissing)
      }
    }
  }

  test("build RDD Watches with Eval") {
    withGpuSparkSession() { spark =>
      import spark.implicits._

      val train = Seq(
        (1.0f, 2.0f, 1.0f, 2.0f, 0.0f, 0.0f),
        (2.0f, 3.0f, 2.0f, 3.0f, 1.0f, 0.1f)
      ).toDF("c1", "c2", "weight", "margin", "label", "other")

      // dataPoint -> (missing, rowNum, nonMissing)
      Map(0.0f -> (0.0f, 5, 9), Float.NaN -> (0.0f, 5, 9)).foreach {
        case (data, (missing, expectedRowNum, expectedNonMissing)) =>
          val eval = Seq(
            (1.0f, 2.0f, 1.0f, 2.0f, 0.0f, 0.0f),
            (2.0f, 3.0f, 2.0f, 3.0f, 1.0f, 0.1f),
            (3.0f, data, 5.0f, 6.0f, 0.0f, 0.1f),
            (4.0f, 5.0f, 6.0f, 7.0f, 0.0f, 0.1f),
            (5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 0.1f)
          ).toDF("c1", "c2", "weight", "margin", "label", "other")

          val features = Array("c1", "c2")
          val classifier = new XGBoostClassifier()
            .setNumWorkers(2)
            .setWeightCol("weight")
            .setBaseMarginCol("margin")
            .setFeaturesCol(features)
            .setDevice("cuda")
            .setMissing(missing)
            .setEvalDataset(eval)

          val rdd = classifier.getPlugin.get.buildRddWatches(classifier, train)
          val result = rdd.mapPartitions { iter =>
            val watches = iter.next()
            val size = watches.size
            val labels = watches.datasets(1).getLabel
            val weight = watches.datasets(1).getWeight
            val margins = watches.datasets(1).getBaseMargin
            val rowNumber = watches.datasets(1).rowNum
            val nonMissing = watches.datasets(1).nonMissingNum
            Iterator.single(size, rowNumber, nonMissing, labels, weight, margins)
          }.collect()

          val labels: ArrayBuffer[Float] = ArrayBuffer.empty
          val weight: ArrayBuffer[Float] = ArrayBuffer.empty
          val margins: ArrayBuffer[Float] = ArrayBuffer.empty
          val rowNumber: ArrayBuffer[Long] = ArrayBuffer.empty
          val nonMissing: ArrayBuffer[Long] = ArrayBuffer.empty

          for (row <- result) {
            assert(row._1 === 2)
            rowNumber.append(row._2)
            nonMissing.append(row._3)
            labels.append(row._4: _*)
            weight.append(row._5: _*)
            margins.append(row._6: _*)
          }
          assert(labels.sorted === Array(0.0f, 1.0f, 0.0f, 0.0f, 1.0f).sorted)
          assert(weight.sorted === Array(1.0f, 2.0f, 5.0f, 6.0f, 7.0f).sorted)
          assert(margins.sorted === Array(2.0f, 3.0f, 6.0f, 7.0f, 8.0f).sorted)
          assert(rowNumber.sum === expectedRowNum)
          assert(nonMissing.sum === expectedNonMissing)
      }
    }
  }


  test("XGBoost-Spark should match xgboost4j") {
    withGpuSparkSession() { spark =>

      val cols = Array("c0", "c1", "c2", "c3", "c4", "c5")
      val label = "label"

      val table = Table.readParquet(new File(getResourcePath("/binary.train.parquet")))
      val df = spark.read.parquet(getResourcePath("/binary.train.parquet"))


      df.show()
    }
  }
}
