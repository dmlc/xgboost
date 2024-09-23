/*
 Copyright (c) 2014-2024 by Contributors

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

import scala.util.Random

trait TrainTestData {

  protected def generateClassificationDataset(
      numRows: Int,
      numClass: Int,
      seed: Int = 1): Seq[(Int, Float, Float, Float, Float)] = {
    val random = new Random()
    random.setSeed(seed)
    (1 to numRows).map { _ =>
      val label = random.nextInt(numClass)
      // label, weight, c1, c2, c3
      (label, random.nextFloat().abs, random.nextGaussian().toFloat, random.nextGaussian().toFloat,
        random.nextGaussian().toFloat)
    }
  }

  protected def generateRegressionDataset(
      numRows: Int,
      seed: Int = 11): Seq[(Float, Float, Float, Float, Float)] = {
    val random = new Random()
    random.setSeed(seed)
    (1 to numRows).map { _ =>
      // label, weight, c1, c2, c3
      (random.nextFloat(), random.nextFloat().abs, random.nextGaussian().toFloat,
        random.nextGaussian().toFloat,
        random.nextGaussian().toFloat)
    }
  }

  protected def generateRankDataset(
      numRows: Int,
      numClass: Int,
      maxGroup: Int = 12,
      seed: Int = 99): Seq[(Int, Float, Int, Float, Float, Float)] = {
    val random = new Random()
    random.setSeed(seed)
    (1 to numRows).map { _ =>
      val group = random.nextInt(maxGroup)
      // label, weight, group, c1, c2, c3
      (random.nextInt(numClass), group.toFloat, group,
        random.nextGaussian().toFloat,
        random.nextGaussian().toFloat,
        random.nextGaussian().toFloat)
    }
  }
}

object Classification extends TrainTestData {
  val train = generateClassificationDataset(300, 2, 3)
  val test = generateClassificationDataset(150, 2, 5)
}

object MultiClassification extends TrainTestData {
  val train = generateClassificationDataset(300, 4, 11)
  val test = generateClassificationDataset(150, 4, 12)
}

object Regression extends TrainTestData {
  val train = generateRegressionDataset(300, 222)
  val test = generateRegressionDataset(150, 223)
}

object Ranking extends TrainTestData {
  val train = generateRankDataset(300, 10, 12, 555)
  val test = generateRankDataset(150, 10, 12, 556)
}
