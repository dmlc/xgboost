/*
 Copyright (c) 2025 by Contributors

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
import java.nio.file.{Files, Paths}
import java.util.concurrent.Executors

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.DurationInt

import ai.rapids.cudf._
import org.apache.commons.logging.LogFactory

import ml.dmlc.xgboost4j.java.{ColumnBatch, CudfColumnBatch}
import ml.dmlc.xgboost4j.scala.spark.Utils.withResource

private[spark] trait ExternalMemory[T] extends Iterator[Table] with AutoCloseable {

  protected val buffers = ArrayBuffer.empty[T]
  private lazy val buffersIterator = buffers.toIterator

  /**
   * Convert the table to T which will be cached
   *
   * @param table to be converted
   * @return the content
   */
  def convertTable(table: Table): T

  /**
   * Load the content to the Table
   *
   * @param content to be loaded
   * @return Table
   */
  def loadTable(content: T): Table

  // Cache the table
  def cacheTable(table: Table): Unit = {
    val content = convertTable(table)
    buffers.append(content)
  }

  override def hasNext: Boolean = buffersIterator.hasNext

  override def next(): Table = loadTable(buffersIterator.next())

  override def close(): Unit = {}
}

// The data will be cached into disk.
private[spark] class DiskExternalMemoryIterator(val parent: String) extends ExternalMemory[String] {

  private val logger = LogFactory.getLog("XGBoostSparkGpuPlugin")

  private lazy val root = {
    val tmp = parent + "/xgboost"
    createDirectory(tmp)
    tmp
  }

  logger.info(s"DiskExternalMemoryIterator createDirectory $root")

  // Tasks mapping the path to the Future of caching table
  private val taskFutures: mutable.HashMap[String, Future[Boolean]] = mutable.HashMap.empty
  private val executor = Executors.newFixedThreadPool(2)
  implicit val ec = ExecutionContext.fromExecutor(executor)

  private var counter = 0

  private def createDirectory(dirPath: String): Unit = {
    val path = Paths.get(dirPath)
    if (!Files.exists(path)) {
      Files.createDirectories(path)
    }
  }

  /**
   * Cache the table into disk which runs in a separate thread
   *
   * @param table to be cached
   * @param path where to cache the table
   */
  private def cacheTableThread(table: Table, path: String): Future[Boolean] = {
    Future {
      withResource(table) { _ =>
        try {
          val names = (1 to table.getNumberOfColumns).map(_.toString)
          val options = ArrowIPCWriterOptions.builder().withColumnNames(names: _*).build()
          withResource(Table.writeArrowIPCChunked(options, new File(path))) { writer =>
            writer.write(table)
          }
          true
        } catch {
          case e: Throwable =>
            throw e
            false
        }
      }
    }
  }

  /**
   * Convert the table to file path which will be cached
   *
   * @param table to be converted
   * @return the content
   */
  override def convertTable(table: Table): String = {
    val path = root + "/table_" + counter + "_" + System.nanoTime()
    counter += 1

    // Increase the reference count of columnars to avoid being recycled
    val newTable = new Table((0 until table.getNumberOfColumns).map(table.getColumn): _*)
    val future = cacheTableThread(newTable, path)
    taskFutures += (path -> future)
    path
  }

  private def closeOnExcept[T <: AutoCloseable, V](r: ArrayBuffer[T])
                                                  (block: ArrayBuffer[T] => V): V = {
    try {
      block(r)
    } catch {
      case t: Throwable =>
        r.foreach(_.close())
        throw t
    }
  }

  private def checkAndWaitCachingDone(path: String): Unit = {
    val futureOpt = taskFutures.get(path)
    if (futureOpt.isEmpty) {
      throw new RuntimeException(s"Failed to find the caching process for $path")
    }
    // Wait 6s to check if the caching is done.
    // TODO, make it configurable
    // If timeout, it's going to throw exception
    val success = Await.result(futureOpt.get, 6.seconds)
    if (!success) { // Failed to cache
      throw new RuntimeException(s"Failed to cache table to $path")
    }
  }

  /**
   * Load the path from disk to the Table
   *
   * @param path to be loaded
   * @return Table
   */
  override def loadTable(path: String): Table = {
    val file = new File(path)

    try {
      checkAndWaitCachingDone(path)

      withResource(Table.readArrowIPCChunked(file)) { reader =>
        val tables = ArrayBuffer.empty[Table]
        closeOnExcept(tables) { tables =>
          var table = Option(reader.getNextIfAvailable())
          while (table.isDefined) {
            tables.append(table.get)
            table = Option(reader.getNextIfAvailable())
          }
        }
        if (tables.size > 1) {
          closeOnExcept(tables) { tables =>
            Table.concatenate(tables.toArray: _*)
          }
        } else {
          tables(0)
        }
      }
    } catch {
      case e: Throwable =>
        close()
        throw e
    } finally {
      if (file.exists()) {
        file.delete()
      }
    }
  }

  override def close(): Unit = {
    executor.shutdown()
    buffers.foreach { path =>
      val file = new File(path)
      if (file.exists()) {
        file.delete()
      }
    }
    buffers.clear()
  }
}

private[spark] object ExternalMemory {
  def apply(path: Option[String] = None): ExternalMemory[_] = {
    path.map(new DiskExternalMemoryIterator(_))
      .getOrElse(throw new RuntimeException("No disk path provided"))
  }
}

/**
 * ExternalMemoryIterator supports iterating the data twice if the `swap` is called.
 *
 * The first round iteration gets the input batch that will be
 *   1. cached in the external memory
 *   2. fed in QuantileDMatrix
 *      The second round iteration returns the cached batch got from external memory.
 *
 * @param input   the spark input iterator
 * @param indices column index
 */
private[scala] class ExternalMemoryIterator(val input: Iterator[Table],
                                            val indices: ColumnIndices,
                                            val path: Option[String] = None)
  extends Iterator[ColumnBatch] {

  private var iter = input

  // Flag to indicate the input has been consumed.
  private var inputIsConsumed = false
  // Flag to indicate the input.next has been called which is valid
  private var inputNextIsCalled = false

  // visible for testing
  private[spark] val externalMemory = ExternalMemory(path)

  override def hasNext: Boolean = {
    val value = iter.hasNext
    if (!value && inputIsConsumed && inputNextIsCalled) {
      externalMemory.close()
    }
    if (!inputIsConsumed && !value && inputNextIsCalled) {
      inputIsConsumed = true
      iter = externalMemory
    }
    value
  }

  override def next(): ColumnBatch = {
    inputNextIsCalled = true
    withResource(new GpuColumnBatch(iter.next())) { batch =>
      if (iter.eq(input)) {
        externalMemory.cacheTable(batch.table)
      }
      new CudfColumnBatch(
        batch.select(indices.featureIds.get),
        batch.select(indices.labelId),
        batch.select(indices.weightId.getOrElse(-1)),
        batch.select(indices.marginId.getOrElse(-1)),
        batch.select(indices.groupId.getOrElse(-1)));
    }
  }
}
