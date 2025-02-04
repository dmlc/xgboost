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

import ai.rapids.cudf.{OrderByArg, Table}
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{Dataset, Row, SparkSession}

import ml.dmlc.xgboost4j.java.CudfColumnBatch
import ml.dmlc.xgboost4j.scala.{DMatrix, QuantileDMatrix, XGBoost => ScalaXGBoost}
import ml.dmlc.xgboost4j.scala.rapids.spark.GpuTestSuite
import ml.dmlc.xgboost4j.scala.rapids.spark.SparkSessionHolder.withSparkSession
import ml.dmlc.xgboost4j.scala.spark.Utils.withResource

class GpuXGBoostPluginSuite extends GpuTestSuite {

  test("params") {
    withGpuSparkSession() { spark =>
      import spark.implicits._
      val df = Seq((1.0f, 2.0f, 1.0f, 2.0f, 0.0f, 0.0f),
        (2.0f, 3.0f, 2.0f, 3.0f, 1.0f, 0.1f),
        (3.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.1f),
        (4.0f, 5.0f, 6.0f, 7.0f, 0.0f, 0.1f),
        (5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.1f)
      ).toDF("c1", "c2", "weight", "margin", "label", "other")
      val xgbParams: Map[String, Any] = Map(
        "max_depth" -> 5,
        "eta" -> 0.2,
        "objective" -> "binary:logistic"
      )
      val features = Array("c1", "c2")
      val estimator = new XGBoostClassifier(xgbParams)
        .setFeaturesCol(features)
        .setMissing(0.2f)
        .setAlpha(0.97)
        .setLeafPredictionCol("leaf")
        .setContribPredictionCol("contrib")
        .setNumRound(3)
        .setDevice("cuda")

      assert(estimator.getMaxDepth === 5)
      assert(estimator.getEta === 0.2)
      assert(estimator.getObjective === "binary:logistic")
      assert(estimator.getFeaturesCols === features)
      assert(estimator.getMissing === 0.2f)
      assert(estimator.getAlpha === 0.97)
      assert(estimator.getDevice === "cuda")
      assert(estimator.getNumRound === 3)

      estimator.setEta(0.66).setMaxDepth(7)
      assert(estimator.getMaxDepth === 7)
      assert(estimator.getEta === 0.66)

      val model = estimator.fit(df)
      assert(model.getMaxDepth === 7)
      assert(model.getEta === 0.66)
      assert(model.getObjective === "binary:logistic")
      assert(model.getFeaturesCols === features)
      assert(model.getMissing === 0.2f)
      assert(model.getAlpha === 0.97)
      assert(model.getLeafPredictionCol === "leaf")
      assert(model.getContribPredictionCol === "contrib")
      assert(model.getDevice === "cuda")
      assert(model.getNumRound === 3)
    }
  }

  test("isEnabled") {
    def checkIsEnabled(spark: SparkSession, expected: Boolean): Unit = {
      import spark.implicits._
      val df = Seq((1.0f, 2.0f, 0.0f),
        (2.0f, 3.0f, 1.0f)
      ).toDF("c1", "c2", "label")
      assert(PluginUtils.getPlugin.isDefined)
      assert(PluginUtils.getPlugin.get.isEnabled(df) === expected)
    }

    // spark.rapids.sql.enabled is not set explicitly, default to true
    withSparkSession(new SparkConf(), spark => {
      checkIsEnabled(spark, expected = true)
    })

    // set spark.rapids.sql.enabled to false
    withCpuSparkSession() { spark =>
      checkIsEnabled(spark, expected = false)
    }

    // set spark.rapids.sql.enabled to true
    withGpuSparkSession() { spark =>
      checkIsEnabled(spark, expected = true)
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

      val plugin = PluginUtils.getPlugin.get.asInstanceOf[GpuXGBoostPlugin]
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
      assert(PluginUtils.getPlugin.isDefined)
      assert(PluginUtils.getPlugin.get.isInstanceOf[GpuXGBoostPlugin])
      var out = PluginUtils.getPlugin.get.asInstanceOf[GpuXGBoostPlugin]
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
      out = PluginUtils.getPlugin.get.asInstanceOf[GpuXGBoostPlugin]
        .preprocess(classifier, df)

      assert(out.schema.names.contains("c1") && out.schema.names.contains("c2"))
      assert(out.schema.names.contains(classifier.getLabelCol))
      assert(out.schema.names.contains("weight") && out.schema.names.contains("margin"))
      assert(out.rdd.getNumPartitions === 4)
    }
  }

  // test distributed
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

          val rdd = PluginUtils.getPlugin.get.buildRddWatches(classifier, df)
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

          val rdd = PluginUtils.getPlugin.get.buildRddWatches(classifier, train)
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

  test("transformed schema") {
    withGpuSparkSession() { spark =>
      import spark.implicits._
      val df = Seq(
        (1.0f, 2.0f, 1.0f, 2.0f, 0.0f, 0.0f),
        (2.0f, 3.0f, 2.0f, 3.0f, 1.0f, 0.1f),
        (3.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.1f),
        (4.0f, 5.0f, 6.0f, 7.0f, 0.0f, 0.1f),
        (5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 0.1f)
      ).toDF("c1", "c2", "weight", "margin", "label", "other")

      val estimator = new XGBoostClassifier()
        .setNumWorkers(1)
        .setNumRound(2)
        .setFeaturesCol(Array("c1", "c2"))
        .setLabelCol("label")
        .setDevice("cuda")

      assert(PluginUtils.getPlugin.isDefined && PluginUtils.getPlugin.get.isEnabled(df))

      val out = estimator.fit(df).transform(df)
      // Transform should not discard the other columns of the transforming dataframe
      Seq("c1", "c2", "weight", "margin", "label", "other").foreach { v =>
        assert(out.schema.names.contains(v))
      }

      // Transform for XGBoostClassifier needs to add extra columns
      Seq("rawPrediction", "probability", "prediction").foreach { v =>
        assert(out.schema.names.contains(v))
      }
      assert(out.schema.names.length === 9)

      val out1 = estimator.setLeafPredictionCol("leaf").setContribPredictionCol("contrib")
        .fit(df)
        .transform(df)
      Seq("leaf", "contrib").foreach { v =>
        assert(out1.schema.names.contains(v))
      }
    }
  }

  private def checkEqual(left: Array[Array[Float]],
                         right: Array[Array[Float]],
                         epsilon: Float = 1e-4f): Unit = {
    assert(left.size === right.size)
    left.zip(right).foreach { case (leftValue, rightValue) =>
      leftValue.zip(rightValue).foreach { case (l, r) =>
        assert(math.abs(l - r) < epsilon)
      }
    }
  }

  Seq("binary:logistic", "multi:softprob").foreach { case objective =>
    test(s"$objective: XGBoost-Spark should match xgboost4j") {
      withGpuSparkSession() { spark =>
        import spark.implicits._

        val numRound = 100
        var xgboostParams: Map[String, Any] = Map(
          "objective" -> objective,
          "device" -> "cuda"
        )

        val (trainPath, testPath) = if (objective == "binary:logistic") {
          (writeFile(Classification.train.toDF("label", "weight", "c1", "c2", "c3")),
            writeFile(Classification.test.toDF("label", "weight", "c1", "c2", "c3")))
        } else {
          xgboostParams = xgboostParams ++ Map("num_class" -> 6)
          (writeFile(MultiClassification.train.toDF("label", "weight", "c1", "c2", "c3")),
            writeFile(MultiClassification.test.toDF("label", "weight", "c1", "c2", "c3")))
        }

        val df = spark.read.parquet(trainPath)
        val testdf = spark.read.parquet(testPath)

        val features = Array("c1", "c2", "c3")
        val featuresIndices = features.map(df.schema.fieldIndex)
        val label = "label"

        val classifier = new XGBoostClassifier(xgboostParams)
          .setFeaturesCol(features)
          .setLabelCol(label)
          .setNumRound(numRound)
          .setLeafPredictionCol("leaf")
          .setContribPredictionCol("contrib")
          .setDevice("cuda")

        val xgb4jModel = withResource(new GpuColumnBatch(
          Table.readParquet(new File(trainPath)))) { batch =>
          val cb = new CudfColumnBatch(batch.select(featuresIndices),
            batch.select(df.schema.fieldIndex(label)), null, null, null
          )
          val qdm = new QuantileDMatrix(Seq(cb).iterator, classifier.getMissing,
            classifier.getMaxBins, classifier.getNthread)
          ScalaXGBoost.train(qdm, xgboostParams, numRound)
        }

        val (xgb4jLeaf, xgb4jContrib, xgb4jProb, xgb4jRaw) = withResource(new GpuColumnBatch(
          Table.readParquet(new File(testPath)))) { batch =>
          val cb = new CudfColumnBatch(batch.select(featuresIndices), null, null, null, null
          )
          val qdm = new DMatrix(cb, classifier.getMissing, classifier.getNthread)
          (xgb4jModel.predictLeaf(qdm), xgb4jModel.predictContrib(qdm),
            xgb4jModel.predict(qdm), xgb4jModel.predict(qdm, outPutMargin = true))
        }

        val rows = classifier.fit(df).transform(testdf).collect()

        // Check Leaf
        val xgbSparkLeaf = rows.map(row => row.getAs[DenseVector]("leaf").toArray.map(_.toFloat))
        checkEqual(xgb4jLeaf, xgbSparkLeaf)

        // Check contrib
        val xgbSparkContrib = rows.map(row =>
          row.getAs[DenseVector]("contrib").toArray.map(_.toFloat))
        checkEqual(xgb4jContrib, xgbSparkContrib)

        // Check probability
        var xgbSparkProb = rows.map(row =>
          row.getAs[DenseVector]("probability").toArray.map(_.toFloat))
        if (objective == "binary:logistic") {
          xgbSparkProb = xgbSparkProb.map(v => Array(v(1)))
        }
        checkEqual(xgb4jProb, xgbSparkProb)

        // Check raw
        var xgbSparkRaw = rows.map(row =>
          row.getAs[DenseVector]("rawPrediction").toArray.map(_.toFloat))
        if (objective == "binary:logistic") {
          xgbSparkRaw = xgbSparkRaw.map(v => Array(v(1)))
        }
        checkEqual(xgb4jRaw, xgbSparkRaw)

      }
    }
  }

  test(s"Regression: XGBoost-Spark should match xgboost4j") {
    withGpuSparkSession() { spark =>
      import spark.implicits._

      val trainPath = writeFile(Regression.train.toDF("label", "weight", "c1", "c2", "c3"))
      val testPath = writeFile(Regression.test.toDF("label", "weight", "c1", "c2", "c3"))

      val df = spark.read.parquet(trainPath)
      val testdf = spark.read.parquet(testPath)

      val features = Array("c1", "c2", "c3")
      val featuresIndices = features.map(df.schema.fieldIndex)
      val label = "label"

      val numRound = 100
      val xgboostParams: Map[String, Any] = Map(
        "device" -> "cuda"
      )

      val regressor = new XGBoostRegressor(xgboostParams)
        .setFeaturesCol(features)
        .setLabelCol(label)
        .setNumRound(numRound)
        .setLeafPredictionCol("leaf")
        .setContribPredictionCol("contrib")
        .setDevice("cuda")

      val xgb4jModel = withResource(new GpuColumnBatch(
        Table.readParquet(new File(trainPath)))) { batch =>
        val cb = new CudfColumnBatch(batch.select(featuresIndices),
          batch.select(df.schema.fieldIndex(label)), null, null, null
        )
        val qdm = new QuantileDMatrix(Seq(cb).iterator, regressor.getMissing,
          regressor.getMaxBins, regressor.getNthread)
        ScalaXGBoost.train(qdm, xgboostParams, numRound)
      }

      val (xgb4jLeaf, xgb4jContrib, xgb4jPred) = withResource(new GpuColumnBatch(
        Table.readParquet(new File(testPath)))) { batch =>
        val cb = new CudfColumnBatch(batch.select(featuresIndices), null, null, null, null
        )
        val qdm = new DMatrix(cb, regressor.getMissing, regressor.getNthread)
        (xgb4jModel.predictLeaf(qdm), xgb4jModel.predictContrib(qdm),
          xgb4jModel.predict(qdm))
      }

      val rows = regressor.fit(df).transform(testdf).collect()

      // Check Leaf
      val xgbSparkLeaf = rows.map(row => row.getAs[DenseVector]("leaf").toArray.map(_.toFloat))
      checkEqual(xgb4jLeaf, xgbSparkLeaf)

      // Check contrib
      val xgbSparkContrib = rows.map(row =>
        row.getAs[DenseVector]("contrib").toArray.map(_.toFloat))
      checkEqual(xgb4jContrib, xgbSparkContrib)

      // Check prediction
      val xgbSparkPred = rows.map(row =>
        Array(row.getAs[Double]("prediction").toFloat))
      checkEqual(xgb4jPred, xgbSparkPred)
    }
  }

  test("The group col should be sorted in each partition") {
    withGpuSparkSession() { spark =>
      import spark.implicits._
      val df = Ranking.train.toDF("label", "weight", "group", "c1", "c2", "c3")

      val xgboostParams: Map[String, Any] = Map(
        "device" -> "cuda",
        "objective" -> "rank:ndcg"
      )
      val features = Array("c1", "c2", "c3")
      val label = "label"
      val group = "group"

      val ranker = new XGBoostRanker(xgboostParams)
        .setFeaturesCol(features)
        .setLabelCol(label)
        .setNumWorkers(1)
        .setNumRound(1)
        .setGroupCol(group)
        .setDevice("cuda")

      val processedDf = PluginUtils.getPlugin.get.asInstanceOf[GpuXGBoostPlugin]
        .preprocess(ranker, df)
      processedDf.rdd.foreachPartition { iter => {
        var prevGroup = Int.MinValue
        while (iter.hasNext) {
          val curr = iter.next()
          val group = curr.asInstanceOf[Row].getAs[Int](1)
          assert(prevGroup <= group)
          prevGroup = group
        }
      }
      }
    }
  }

  test("Same group must be in the same partition") {
    val num_workers = 3
    withGpuSparkSession() { spark =>
      import spark.implicits._
      val df = spark.createDataFrame(spark.sparkContext.parallelize(Seq(
        (0.1, 1, 0),
        (0.1, 1, 0),
        (0.1, 1, 0),
        (0.1, 1, 1),
        (0.1, 1, 1),
        (0.1, 1, 1),
        (0.1, 1, 2),
        (0.1, 1, 2),
        (0.1, 1, 2)), 1)).toDF("label", "f1", "group")

      // The original pattern will repartition df in a RoundRobin manner
      val oriRows = df.repartition(num_workers)
        .sortWithinPartitions(df.col("group"))
        .select("group")
        .mapPartitions { case iter =>
          val tmp: ArrayBuffer[Int] = ArrayBuffer.empty
          while (iter.hasNext) {
            val r = iter.next()
            tmp.append(r.getInt(0))
          }
          Iterator.single(tmp.mkString(","))
        }.collect()
      assert(oriRows.length == 3)
      assert(oriRows.contains("0,1,2"))

      // The fix has replaced repartition with repartitionByRange which will put the
      // instances with same group into the same partition
      val ranker = new XGBoostRanker().setGroupCol("group").setNumWorkers(num_workers)
      val processedDf = PluginUtils.getPlugin.get.asInstanceOf[GpuXGBoostPlugin]
        .preprocess(ranker, df)
      val rows = processedDf
        .select("group")
        .mapPartitions { case iter =>
          val tmp: ArrayBuffer[Int] = ArrayBuffer.empty
          while (iter.hasNext) {
            val r = iter.next()
            tmp.append(r.getInt(0))
          }
          Iterator.single(tmp.mkString(","))
        }.collect()

      rows.forall(Seq("0,0,0", "1,1,1", "2,2,2").contains)
    }
  }

  test("Ranker: XGBoost-Spark should match xgboost4j") {
    withGpuSparkSession() { spark =>
      import spark.implicits._

      val trainPath = writeFile(Ranking.train.toDF("label", "weight", "group", "c1", "c2", "c3"))
      val testPath = writeFile(Ranking.test.toDF("label", "weight", "group", "c1", "c2", "c3"))

      val df = spark.read.parquet(trainPath)
      val testdf = spark.read.parquet(testPath)

      val features = Array("c1", "c2", "c3")
      val featuresIndices = features.map(df.schema.fieldIndex)
      val label = "label"
      val group = "group"

      val numRound = 100
      val xgboostParams: Map[String, Any] = Map(
        "device" -> "cuda",
        "objective" -> "rank:ndcg"
      )

      val ranker = new XGBoostRanker(xgboostParams)
        .setFeaturesCol(features)
        .setLabelCol(label)
        .setNumRound(numRound)
        .setLeafPredictionCol("leaf")
        .setContribPredictionCol("contrib")
        .setGroupCol(group)
        .setDevice("cuda")

      val xgb4jModel = withResource(new GpuColumnBatch(
        Table.readParquet(new File(trainPath)
        ).orderBy(OrderByArg.asc(df.schema.fieldIndex(group))))) { batch =>
        val cb = new CudfColumnBatch(batch.select(featuresIndices),
          batch.select(df.schema.fieldIndex(label)), null, null,
          batch.select(df.schema.fieldIndex(group)))
        val qdm = new QuantileDMatrix(Seq(cb).iterator, ranker.getMissing,
          ranker.getMaxBins, ranker.getNthread)
        ScalaXGBoost.train(qdm, xgboostParams, numRound)
      }

      val (xgb4jLeaf, xgb4jContrib, xgb4jPred) = withResource(new GpuColumnBatch(
        Table.readParquet(new File(testPath)))) { batch =>
        val cb = new CudfColumnBatch(batch.select(featuresIndices), null, null, null, null
        )
        val qdm = new DMatrix(cb, ranker.getMissing, ranker.getNthread)
        (xgb4jModel.predictLeaf(qdm), xgb4jModel.predictContrib(qdm),
          xgb4jModel.predict(qdm))
      }

      val rows = ranker.fit(df).transform(testdf).collect()

      // Check Leaf
      val xgbSparkLeaf = rows.map(row => row.getAs[DenseVector]("leaf").toArray.map(_.toFloat))
      checkEqual(xgb4jLeaf, xgbSparkLeaf)

      // Check contrib
      val xgbSparkContrib = rows.map(row =>
        row.getAs[DenseVector]("contrib").toArray.map(_.toFloat))
      checkEqual(xgb4jContrib, xgbSparkContrib)

      // Check prediction
      val xgbSparkPred = rows.map(row =>
        Array(row.getAs[Double]("prediction").toFloat))
      checkEqual(xgb4jPred, xgbSparkPred)
    }
  }

  def writeFile(df: Dataset[_]): String = {
    def listFiles(directory: String): Array[String] = {
      val dir = new File(directory)
      if (dir.exists && dir.isDirectory) {
        dir.listFiles.filter(f => f.isFile && f.getName.startsWith("part-")).map(_.getName)
      } else {
        Array.empty[String]
      }
    }

    val dir = createTmpFolder("gpu_").toAbsolutePath.toString
    df.coalesce(1).write.parquet(s"$dir/data")

    val file = listFiles(s"$dir/data")(0)
    s"$dir/data/$file"
  }

}
