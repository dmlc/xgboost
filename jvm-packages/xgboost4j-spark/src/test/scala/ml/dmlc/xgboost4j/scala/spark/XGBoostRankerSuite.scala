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

import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.funsuite.AnyFunSuite

import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => ScalaXGBoost}
import ml.dmlc.xgboost4j.scala.spark.Regression.Ranking
import ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams.RANKER_OBJS
import ml.dmlc.xgboost4j.scala.spark.params.XGBoostParams

class XGBoostRankerSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

  test("XGBoostRanker copy") {
    val ranker = new XGBoostRanker().setNthread(2).setNumWorkers(10)
    val rankertCopied = ranker.copy(ParamMap.empty)

    assert(ranker.uid === rankertCopied.uid)
    assert(ranker.getNthread === rankertCopied.getNthread)
    assert(ranker.getNumWorkers === ranker.getNumWorkers)
  }

  test("XGBoostRankerModel copy") {
    val model = new XGBoostRankerModel("hello").setNthread(2).setNumWorkers(10)
    val modelCopied = model.copy(ParamMap.empty)
    assert(model.uid === modelCopied.uid)
    assert(model.getNthread === modelCopied.getNthread)
    assert(model.getNumWorkers === modelCopied.getNumWorkers)
  }

  test("read/write") {
    val trainDf = smallGroupVector
    val xgbParams: Map[String, Any] = Map(
      "max_depth" -> 5,
      "eta" -> 0.2,
      "objective" -> "rank:ndcg"
    )

    def check(xgboostParams: XGBoostParams[_]): Unit = {
      assert(xgboostParams.getMaxDepth === 5)
      assert(xgboostParams.getEta === 0.2)
      assert(xgboostParams.getObjective === "rank:ndcg")
    }

    val rankerPath = new File(tempDir.toFile, "ranker").getPath
    val ranker = new XGBoostRanker(xgbParams).setNumRound(1).setGroupCol("group")
    check(ranker)
    assert(ranker.getGroupCol === "group")

    ranker.write.overwrite().save(rankerPath)
    val loadedRanker = XGBoostRanker.load(rankerPath)
    check(loadedRanker)
    assert(loadedRanker.getGroupCol === "group")

    val model = loadedRanker.fit(trainDf)
    check(model)
    assert(model.getGroupCol === "group")

    val modelPath = new File(tempDir.toFile, "model").getPath
    model.write.overwrite().save(modelPath)
    val modelLoaded = XGBoostRankerModel.load(modelPath)
    check(modelLoaded)
    assert(modelLoaded.getGroupCol === "group")
  }

  test("validate") {
    val trainDf = smallGroupVector
    val ranker = new XGBoostRanker()
    // must define group column
    intercept[IllegalArgumentException](
      ranker.validate(trainDf)
    )
    val ranker1 = new XGBoostRanker().setGroupCol("group")
    ranker1.validate(trainDf)
    assert(ranker1.getObjective === "rank:ndcg")
  }

  test("XGBoostRankerModel transformed schema") {
    val trainDf = smallGroupVector
    val ranker = new XGBoostRanker().setGroupCol("group").setNumRound(1)
    val model = ranker.fit(trainDf)
    var out = model.transform(trainDf)
    // Transform should not discard the other columns of the transforming dataframe
    Seq("label", "group", "margin", "weight", "features").foreach { v =>
      assert(out.schema.names.contains(v))
    }
    // Ranker does not have extra columns
    Seq("rawPrediction", "probability").foreach { v =>
      assert(!out.schema.names.contains(v))
    }
    assert(out.schema.names.contains("prediction"))
    assert(out.schema.names.length === 6)
    model.setLeafPredictionCol("leaf").setContribPredictionCol("contrib")
    out = model.transform(trainDf)
    assert(out.schema.names.contains("leaf"))
    assert(out.schema.names.contains("contrib"))
  }

  test("Supported objectives") {
    val ranker = new XGBoostRanker().setGroupCol("group")
    val df = smallGroupVector
    RANKER_OBJS.foreach { obj =>
      ranker.setObjective(obj)
      ranker.validate(df)
    }

    ranker.setObjective("binary:logistic")
    intercept[IllegalArgumentException](
      ranker.validate(df)
    )
  }

  test("The group col should be sorted in each partition") {
    val trainingDF = buildDataFrameWithGroup(Ranking.train)

    val ranker = new XGBoostRanker()
      .setNumRound(1)
      .setNumWorkers(numWorkers)
      .setGroupCol("group")

    val (df, _) = ranker.preprocess(trainingDF)
    df.rdd.foreachPartition { iter => {
      var prevGroup = Int.MinValue
      while (iter.hasNext) {
        val curr = iter.next()
        val group = curr.asInstanceOf[Row].getAs[Int](2)
        assert(prevGroup <= group)
        prevGroup = group
      }
    }}
  }

  test("Same group must be in the same partition") {
    val spark = ss
    import spark.implicits._
    val num_workers = 3
    val df = ss.createDataFrame(sc.parallelize(Seq(
      (0.1, Vectors.dense(1.0, 2.0, 3.0), 0),
      (0.1, Vectors.dense(0.0, 0.0, 0.0), 0),
      (0.1, Vectors.dense(0.0, 3.0, 0.0), 0),
      (0.1, Vectors.dense(2.0, 0.0, 4.0), 1),
      (0.1, Vectors.dense(0.2, 1.2, 2.0), 1),
      (0.1, Vectors.dense(0.5, 2.2, 1.7), 1),
      (0.1, Vectors.dense(0.5, 2.2, 1.7), 2),
      (0.1, Vectors.dense(0.5, 2.2, 1.7), 2),
      (0.1, Vectors.dense(0.5, 2.2, 1.7), 2)), 1)).toDF("label", "features", "group")

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
    val (processedDf, _) = ranker.preprocess(df)
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

  private def runLengthEncode(input: Seq[Int]): Seq[Int] = {
    if (input.isEmpty) return Seq(0)

    input.indices
      .filter(i => i == 0 || input(i) != input(i - 1)) :+ input.length
  }

  private def runRanker(ranker: XGBoostRanker, dataset: Dataset[_]): (Array[Float], Array[Int]) = {
    val (df, indices) = ranker.preprocess(dataset)
    val rdd = ranker.toRdd(df, indices)
    val result = rdd.mapPartitions { iter =>
      if (iter.hasNext) {
        val watches = iter.next()
        val dm = watches.toMap(Utils.TRAIN_NAME)
        val weight = dm.getWeight
        val group = dm.getGroup
        watches.delete()
        Iterator.single((weight, group))
      } else {
        Iterator.empty
      }
    }.collect()

    val weight: ArrayBuffer[Float] = ArrayBuffer.empty
    val group: ArrayBuffer[Int] = ArrayBuffer.empty

    for (row <- result) {
      weight.append(row._1: _*)
      group.append(row._2: _*)
    }
    (weight.toArray, group.toArray)
  }

  Seq(None, Some("weight")).foreach { weightCol => {
    val msg = weightCol.map(_ => "with weight").getOrElse("without weight")
    test(s"to RDD watches with group $msg") {
      // One instance without setting weight
      var df = ss.createDataFrame(sc.parallelize(Seq(
        (1.0, 0, 10, Vectors.dense(Array(1.0, 2.0, 3.0)))
      ))).toDF("label", "group", "weight", "features")

      val ranker = new XGBoostRanker()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setGroupCol("group")
        .setNumWorkers(1)

      weightCol.foreach(ranker.setWeightCol)

      val (weights, groupSize) = runRanker(ranker, df)
      val expectedWeight = weightCol.map(_ => Array(10.0f)).getOrElse(Array(1.0f))
      assert(weights === expectedWeight)
      assert(groupSize === runLengthEncode(Seq(0)))

      df = ss.createDataFrame(sc.parallelize(Seq(
        (1.0, 1, 2, Vectors.dense(Array(1.0, 2.0, 3.0))),
        (2.0, 1, 2, Vectors.dense(Array(1.0, 2.0, 3.0))),
        (1.0, 0, 5, Vectors.dense(Array(1.0, 2.0, 3.0))),
        (0.0, 1, 2, Vectors.dense(Array(1.0, 2.0, 3.0))),
        (1.0, 0, 5, Vectors.dense(Array(1.0, 2.0, 3.0))),
        (2.0, 2, 7, Vectors.dense(Array(1.0, 2.0, 3.0)))
      ))).toDF("label", "group", "weight", "features")

      val groups = Array(1, 1, 0, 1, 0, 2).sorted
      val (weights1, groupSize1) = runRanker(ranker, df)
      val expectedWeight1 = weightCol.map(_ => Array(5.0f, 2.0f, 7.0f))
        .getOrElse(groups.distinct.map(_ => 1.0f))

      assert(groupSize1 === runLengthEncode(groups))
      assert(weights1 === expectedWeight1)
    }
  }
  }

  test("XGBoost-Spark output should match XGBoost4j") {
    val trainingDM = new DMatrix(Ranking.train.iterator)
    val weights = Ranking.trainGroups.distinct.map(_ => 1.0f).toArray
    trainingDM.setQueryId(Ranking.trainGroups.toArray)
    trainingDM.setWeight(weights)

    val testDM = new DMatrix(Ranking.test.iterator)
    val trainingDF = buildDataFrameWithGroup(Ranking.train)
    val testDF = buildDataFrameWithGroup(Ranking.test)
    val paramMap = Map("objective" -> "rank:ndcg")
    checkResultsWithXGBoost4j(trainingDM, testDM, trainingDF, testDF, 5, paramMap)
  }

  test("XGBoost-Spark output with weight should match XGBoost4j") {
    val trainingDM = new DMatrix(Ranking.trainWithWeight.iterator)
    trainingDM.setQueryId(Ranking.trainGroups.toArray)
    trainingDM.setWeight(Ranking.trainGroups.distinct.map(_.toFloat).toArray)

    val testDM = new DMatrix(Ranking.test.iterator)
    val trainingDF = buildDataFrameWithGroup(Ranking.trainWithWeight)
    val testDF = buildDataFrameWithGroup(Ranking.test)
    val paramMap = Map("objective" -> "rank:ndcg")
    checkResultsWithXGBoost4j(trainingDM, testDM, trainingDF, testDF,
      5, paramMap, Some("weight"))
  }

  private def checkResultsWithXGBoost4j(
      trainingDM: DMatrix,
      testDM: DMatrix,
      trainingDF: DataFrame,
      testDF: DataFrame,
      round: Int = 5,
      xgbParams: Map[String, Any] = Map.empty,
      weightCol: Option[String] = None): Unit = {
    val paramMap = Map(
      "eta" -> "1",
      "max_depth" -> "6",
      "base_score" -> 0.5,
      "max_bin" -> 16) ++ xgbParams
    val xgb4jModel = ScalaXGBoost.train(trainingDM, paramMap, round)

    val ranker = new XGBoostRanker(paramMap)
      .setNumRound(round)
      // If we use multi workers to train the ranking, the result probably will be different
      .setNumWorkers(1)
      .setLeafPredictionCol("leaf")
      .setContribPredictionCol("contrib")
      .setGroupCol("group")
    weightCol.foreach(weight => ranker.setWeightCol(weight))

    def checkEqual(left: Array[Array[Float]], right: Map[Int, Array[Float]]) = {
      assert(left.size === right.size)
      left.zipWithIndex.foreach { case (leftValue, index) =>
        assert(leftValue.sameElements(right(index)))
      }
    }

    val xgbSparkModel = ranker.fit(trainingDF)
    val rows = xgbSparkModel.transform(testDF).collect()

    // Check Leaf
    val xgb4jLeaf = xgb4jModel.predictLeaf(testDM)
    val xgbSparkLeaf = rows.map(row =>
      (row.getAs[Int]("id"), row.getAs[DenseVector]("leaf").toArray.map(_.toFloat))).toMap
    checkEqual(xgb4jLeaf, xgbSparkLeaf)

    // Check contrib
    val xgb4jContrib = xgb4jModel.predictContrib(testDM)
    val xgbSparkContrib = rows.map(row =>
      (row.getAs[Int]("id"), row.getAs[DenseVector]("contrib").toArray.map(_.toFloat))).toMap
    checkEqual(xgb4jContrib, xgbSparkContrib)

    // Check prediction
    val xgb4jPred = xgb4jModel.predict(testDM)
    val xgbSparkPred = rows.map(row => {
      val pred = row.getAs[Double]("prediction").toFloat
      (row.getAs[Int]("id"), Array(pred))
    }).toMap
    checkEqual(xgb4jPred, xgbSparkPred)
  }

}
