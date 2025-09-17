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

import java.nio.file.{Files, Path}

import org.apache.spark.network.util.JavaUtils
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

trait TmpFolderPerSuite extends BeforeAndAfterAll { self: AnyFunSuite =>
  protected var tempDir: Path = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    tempDir = Files.createTempDirectory(getClass.getName)
  }

  override def afterAll(): Unit = {
    JavaUtils.deleteRecursively(tempDir.toFile)
    super.afterAll()
  }

  protected def createTmpFolder(prefix: String): Path = {
    Files.createTempDirectory(tempDir, prefix)
  }

}
