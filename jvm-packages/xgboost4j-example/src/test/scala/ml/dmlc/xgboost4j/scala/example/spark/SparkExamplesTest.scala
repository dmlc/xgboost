/*
 Copyright (c) 2014-2023 by Contributors

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
package ml.dmlc.xgboost4j.scala.example.spark

import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.slf4j.LoggerFactory

import java.io.File
import java.nio.file.{Files, StandardOpenOption}
import scala.jdk.CollectionConverters._
import scala.util.{Random, Try}

class SparkExamplesTest extends AnyFunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[SparkExamplesTest])
  private val random = new Random(42)
  protected val numWorkers: Int = scala.math.min(Runtime.getRuntime.availableProcessors(), 4)

  private val pathToTestDataset = Files.createTempFile("", "iris.csv").toAbsolutePath
  private var spark: SparkSession = _

  override def beforeAll(): Unit = {

    def generateLine(i: Int): String = {
      val getIrisName = (int: Int) => {
        int % 3 match {
          case 0 => "Iris-versicolor"
          case 1 => "Iris-virginica"
          case 2 => "Iris-setosa"
        }
      }
      val generateValue = () => Math.abs(random.nextInt(99) * 0.1)
      val sepalLength = generateValue()
      val sepalWidth = generateValue()
      val petalLength = generateValue()
      val petalWidth = generateValue()
      val irisName = getIrisName(Math.abs(random.nextInt()) + i)
      s"$sepalLength,$sepalWidth,$petalLength,$petalWidth,$irisName"
    }

    if (spark == null) {
       spark = SparkSession
        .builder()
        .appName("XGBoost4J-Spark Pipeline Example")
        .master(s"local[${numWorkers}]")
        .config("spark.ui.enabled", value = false)
        .config("spark.driver.memory", "512m")
        .config("spark.barrier.sync.timeout", 10)
        .config("spark.task.cpus", 1)
        .getOrCreate()
      spark.sparkContext.setLogLevel("ERROR")
    }
    val data = (0 until 150)
      .map(i => generateLine(i))
      .toList
      .asJava
    Files.write(pathToTestDataset,
      data,
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE,
      StandardOpenOption.TRUNCATE_EXISTING)
    logger.info(s"${new String(Files.readAllBytes(pathToTestDataset))}")

  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
      cleanExternalCache(spark.sparkContext.appName)
      spark = null
    }

    Try(Files.deleteIfExists(pathToTestDataset))
      .recover {
        case e =>
          logger.warn(
            s"Could not delete temporary file $pathToTestDataset. Please, remove it manually",
            e
          )
          true
    }
  }

  private def cleanExternalCache(prefix: String): Unit = {
    val dir = new File(".")
    for (file <- dir.listFiles() if file.getName.startsWith(prefix)) {
      file.delete()
    }
  }

  test("Smoke test for SparkMLlibPipeline example") {
    SparkMLlibPipeline.run(spark, pathToTestDataset.toString, "target/native-model",
      "target/pipeline-model", "auto", 2)
  }

  test("Smoke test for SparkTraining example") {
    val spark = SparkSession
      .builder()
      .appName("XGBoost4J-Spark Pipeline Example")
      .master(s"local[${numWorkers}]")
      .config("spark.ui.enabled", value = false)
      .config("spark.driver.memory", "512m")
      .config("spark.barrier.sync.timeout", 10)
      .config("spark.task.cpus", 1)
      .getOrCreate()

    SparkTraining.run(spark, pathToTestDataset.toString, "auto", 2)
  }
}
