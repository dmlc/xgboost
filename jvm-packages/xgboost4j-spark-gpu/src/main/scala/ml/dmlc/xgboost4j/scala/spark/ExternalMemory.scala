/*
 Copyright (c) 2021-2024 by Contributors

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

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf._

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

case class HostMemoryBufferInfo(hostMemoryBuffer: HostMemoryBuffer, size: Long)

// The data will be cached into host memory
private[spark] class HostExternalMemoryIterator()
  extends ExternalMemory[HostMemoryBufferInfo] {

  private lazy val allocator = DefaultHostMemoryAllocator.get()

  class XGBoostHostBufferConsumer extends HostBufferConsumer {
    private val hostBuffers = ArrayBuffer.empty[HostMemoryBufferInfo]

    override def handleBuffer(hostMemoryBuffer: HostMemoryBuffer, l: Long): Unit = {
      hostBuffers.append(HostMemoryBufferInfo(hostMemoryBuffer = hostMemoryBuffer, size = l))
    }

    def getHostMemoryBuffer: HostMemoryBufferInfo = {
      if (hostBuffers.size == 1) {
        hostBuffers(0)
      } else if (hostBuffers.size > 1) {
        val totalSize = hostBuffers.map(_.size).sum
        val buffer = allocator.allocate(totalSize)
        var offset = 0L

        hostBuffers.foreach { h =>
          withResource(h.hostMemoryBuffer) { _ =>
            buffer.copyFromHostBuffer(offset, h.hostMemoryBuffer, 0, h.size)
            offset += h.size
          }
        }
        HostMemoryBufferInfo(buffer, totalSize)
      } else {
        throw new RuntimeException("No data") // Unreachable
      }
    }
  }

  /**
   * Convert the table to T which will be cached
   *
   * @param table to be converted
   * @return the content
   */
  override def convertTable(table: Table): HostMemoryBufferInfo = {
    val names = (0 until table.getNumberOfColumns).map(_.toString)
    val options = ArrowIPCWriterOptions.builder().withNotNullableColumnNames(names: _*).build()
    val consumer = new XGBoostHostBufferConsumer()
    withResource(Table.writeArrowIPCChunked(options, consumer)) { writer =>
      writer.write(table)
    }
    consumer.getHostMemoryBuffer
  }

  class XGBoostHostBufferProvider(bufferInfo: HostMemoryBufferInfo) extends HostBufferProvider {
    var offset = 0L

    override def readInto(hostMemoryBuffer: HostMemoryBuffer, l: Long): Long = {
      val amountLeft = bufferInfo.size - offset
      val amountToCopy = Math.max(0, Math.min(l, amountLeft))
      if (amountToCopy > 0) {
        hostMemoryBuffer.copyFromHostBuffer(0, bufferInfo.hostMemoryBuffer, offset, amountToCopy)
        offset += amountToCopy
      }
      amountToCopy
    }

    override def close(): Unit = {
      //      bufferInfo.hostMemoryBuffer.close()
    }
  }

  /**
   * Load the content to the Table
   *
   * @param content to be loaded
   * @return Table
   */
  override def loadTable(content: HostMemoryBufferInfo): Table = {
    val tables = ArrayBuffer.empty[Table]
    withResource(new XGBoostHostBufferProvider(content)) { provider =>
      withResource(Table.readArrowIPCChunked(provider)) { reader =>
        var table = Option(reader.getNextIfAvailable)
        while (table.isDefined) {
          tables.append(table.get)
          table = Option(reader.getNextIfAvailable)
        }
        if (tables.size == 1) {
          tables(0)
        } else if (tables.size > 1) {
          withResource(tables) { _ =>
            Table.concatenate(tables: _*)
          }
        } else {
          throw new RuntimeException("No data") // Unreachable
        }
      }
    }
  }
}

// The data will be cached into disk.
private[spark] class DiskExternalMemoryIterator(val path: String) extends ExternalMemory[String] {

  private lazy val root = {
    val tmp = path + "/xgboost"
    createDirectory(tmp)
    tmp
  }

  private var counter = 0

  private def createDirectory(dirPath: String): Unit = {
    val path = Paths.get(dirPath)
    if (!Files.exists(path)) {
      Files.createDirectories(path)
      println(s"Directory created: $dirPath")
    } else {
      println(s"Directory already exists: $dirPath")
    }
  }

  /**
   * Convert the table to file path which will be cached
   *
   * @param table to be converted
   * @return the content
   */
  override def convertTable(table: Table): String = {
    val names = (1 to table.getNumberOfColumns).map(_.toString)
    val options = ArrowIPCWriterOptions.builder().withColumnNames(names: _*).build()
    val path = root + "/table_" + counter + "_" + System.nanoTime();
    counter += 1
    withResource(Table.writeArrowIPCChunked(options, new File(path))) { writer =>
      writer.write(table)
    }
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

  /**
   * Load the path from disk to the Table
   *
   * @param content to be loaded
   * @return Table
   */
  override def loadTable(content: String): Table = {

    withResource(Table.readArrowIPCChunked(new File(content))) { reader =>
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
  }

  override def close(): Unit = {
    buffers.foreach { path =>
      val file = new File(path)
      if (file.exists()) {
        file.delete()
      }
    }
  }
}

private[spark] object ExternalMemory {
  def apply(path: Option[String] = None): ExternalMemory[_] = {
    path.map(new DiskExternalMemoryIterator(_)).getOrElse(new HostExternalMemoryIterator())
  }
}

/**
 * ExternalMemoryIterator supports iterating the data twice if the `swap` is called.
 *
 * The first round iteration gets the input batch that will be
 *   1. cached in the external memory
 *      2. fed in QuantilDmatrix
 *      The second round iteration returns the cached batch got from external memory.
 *
 * @param input   the spark input iterator
 * @param indices column index
 */
private[spark] class ExternalMemoryIterator(val input: Iterator[Table],
                                            val indices: ColumnIndices,
                                            val path: Option[String] = None)
  extends Iterator[ColumnBatch] with AutoCloseable {

  private var iter = input

  // Flag to indicate the input has been consumed.
  private var inputIsConsumed = false
  // Flag to indicate the input.next has been called which is valid
  private var inputNextIsValid = false

  // visible for testing
  private[spark] val externalMemory = ExternalMemory(path)

  override def hasNext: Boolean = {
    val value = iter.hasNext
    if (!inputIsConsumed && !value && inputNextIsValid) {
      inputIsConsumed = true
      iter = externalMemory
    }
    value
  }

  override def next(): ColumnBatch = {
    inputNextIsValid = true
    withResource(new GpuColumnBatch(iter.next())) { batch =>
      if (iter == input) {
        externalMemory.cacheTable(batch.getTable())
      }

      new CudfColumnBatch(
        batch.select(indices.featureIds.get),
        batch.select(indices.labelId),
        batch.select(indices.weightId.getOrElse(-1)),
        batch.select(indices.marginId.getOrElse(-1)),
        batch.select(indices.groupId.getOrElse(-1)));
    }
  }

  override def close(): Unit = externalMemory.close()
}
