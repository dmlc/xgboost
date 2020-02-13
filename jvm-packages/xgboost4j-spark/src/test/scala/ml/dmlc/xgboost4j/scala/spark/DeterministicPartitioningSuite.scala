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

import org.scalatest.FunSuite

import org.apache.spark.sql.functions._

class DeterministicPartitioningSuite extends FunSuite with TmpFolderPerSuite with PerTest {

  test("perform deterministic partitioning when checkpointInternal and" +
    " checkpointPath is set (Classifier)") {
    val tmpPath = createTmpFolder("model1").toAbsolutePath.toString
    val paramMap = Map("eta" -> "1", "max_depth" -> 2,
      "objective" -> "binary:logistic", "checkpoint_path" -> tmpPath,
      "checkpoint_interval" -> 2, "num_workers" -> numWorkers)
    val xgbClassifier = new XGBoostClassifier(paramMap)
    assert(xgbClassifier.needDeterministicRepartitioning)
  }

  test("perform deterministic partitioning when checkpointInternal and" +
    " checkpointPath is set (Regressor)") {
    val tmpPath = createTmpFolder("model1").toAbsolutePath.toString
    val paramMap = Map("eta" -> "1", "max_depth" -> 2,
      "objective" -> "binary:logistic", "checkpoint_path" -> tmpPath,
      "checkpoint_interval" -> 2, "num_workers" -> numWorkers)
    val xgbRegressor = new XGBoostRegressor(paramMap)
    assert(xgbRegressor.needDeterministicRepartitioning)
  }

  test("deterministic partitioning takes effect with various parts of data") {
    val trainingDF = buildDataFrame(Classification.train)
    // the test idea is that, we apply a chain of repartitions over trainingDFs but they
    // have to produce the identical RDDs
    val transformedDFs = (1 until 6).map(shuffleCount => {
      var resultDF = trainingDF
      for (i <- 0 until shuffleCount) {
        resultDF = resultDF.repartition(numWorkers)
      }
      resultDF
    })
    val transformedRDDs = transformedDFs.map(df => DataUtils.convertDataFrameToXGBLabeledPointRDDs(
      col("label"),
      col("features"),
      lit(1.0),
      lit(Float.NaN),
      None,
      numWorkers,
      deterministicPartition = true,
      df
    ).head)
    val resultsMaps = transformedRDDs.map(rdd => rdd.mapPartitionsWithIndex {
      case (partitionIndex, labelPoints) =>
        Iterator((partitionIndex, labelPoints.toList))
    }.collect().toMap)
    resultsMaps.foldLeft(resultsMaps.head) { case (map1, map2) =>
      assert(map1.keys.toSet === map2.keys.toSet)
      for ((parIdx, labeledPoints) <- map1) {
        val sortedA = labeledPoints.sortBy(_.hashCode())
        val sortedB = map2(parIdx).sortBy(_.hashCode())
        assert(sortedA.length === sortedB.length)
        assert(sortedA.indices.forall(idx =>
          sortedA(idx).values.toSet === sortedB(idx).values.toSet))
      }
      map2
    }
  }
}
