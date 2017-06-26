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

import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, BeforeAndAfterAll, FunSuite}

trait SharedSparkContext extends FunSuite with BeforeAndAfter with BeforeAndAfterAll
  with Serializable {

  @transient protected implicit var sc: SparkContext = _

  override def beforeAll() {
    val sparkConf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("XGBoostSuite")
      .set("spark.driver.memory", "512m")
      .set("spark.ui.enabled", "false")

    sc = new SparkContext(sparkConf)
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
      sc = null
    }
  }
}
