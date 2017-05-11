package com.airbnb.common.ml.util

import java.io._
import java.net.URI
import java.util.GregorianCalendar

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.hadoop.io.compress.{CompressionCodec, GzipCodec}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{Accumulator, SparkContext}
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}
import org.joda.time.{DateTime, Days}


object PipelineUtil extends ScalaLogging {
  val hadoopConfiguration = new Configuration()
  val defaultCodec = classOf[GzipCodec]

  def saveAndCommit(saveRDD: String => Unit, dest: String, overwrite: Boolean = false): Unit = {
    logger.info("Saving data to %s".format(dest))

    val hfs = FileSystem.get(new java.net.URI(dest), new Configuration())

    val tmp = dest + ".tmp"
    val tmpPath = new Path(tmp)
    val destPath = new Path(dest)

    if (!hfs.exists(destPath) || overwrite) {
      try {
        if (hfs.exists(tmpPath)) {
          hfs.delete(tmpPath, true)
          logger.info("deleted old tmp directory: " + tmpPath)
        }
        saveRDD(tmp)
        if (hfs.exists(destPath)) {
          hfs.delete(destPath, true)
          logger.info("deleted old directory: " + destPath)
        }
        logger.info("committing " + dest)
        hfs.rename(tmpPath, destPath)
        logger.info("committed " + dest)
      } catch {
        case e: Exception =>
          logger.error("exception during save: ", e)
          logger.info("deleting failed data for " + dest + ".tmp")
          hfs.delete(tmpPath, true)
          logger.info("deleted " + dest + ".tmp")
          throw e
      }
    } else {
      logger.info("data already exists for " + dest)
    }
  }

  // Save result to output as textfile and update outputTable as partitionSpec
  // This function only works for external table!
  def saveToHdfsAndUpdateHive[U](hc: HiveContext,
      output: String,
      outputTable: String,
      partitionSpec: String,
      result: RDD[U],
      overwrite: Boolean):Unit = {
    PipelineUtil.saveAndCommitAsTextFile(result, output, overwrite)
    HiveUtil.updateHivePartition(hc, outputTable, partitionSpec, output)
  }

  def saveAndCommitAsTextFile[U](rdd: RDD[U], dest: String): Unit = {
    saveAndCommitAsTextFile(rdd, dest, 0)
  }

  def saveAndCommitAsTextFile[U](rdd: RDD[U], dest: String, overwrite: Boolean): Unit = {
    saveAndCommitAsTextFile(rdd, dest, 0, overwrite)
  }

  def saveAndCommitAsTextFile[U](
      rdd: RDD[U],
      dest: String,
      partition: Int,
      overwrite: Boolean = false,
      codec: Class[_ <: CompressionCodec] = defaultCodec): Unit = {
    saveAndCommit(output => {
      if (partition > 0) {
        // shuffle it to prevent imbalance output, which slows down the whole job.
        rdd.coalesce(partition, true)
          .saveAsTextFile(output, codec)
      } else {
        rdd.saveAsTextFile(output, codec)
      }
    }, dest, overwrite)
  }

  /*
   * Given an RDD of Tuple (or case class) type rows,
   * save it to a partitioned Hive table.
   * The columns must be in the same order as the Tuple elements
   * (or the fields in the case class).
   */
  def saveRDDToManagedPartitionedHiveTable[T <: Product : TypeTag](
      sc: SparkContext,
      rowsToSave: RDD[T],
      columnNames: Seq[String],
      hiveTableName: String,
      hivePartitionSpecs: Map[String, Any]
  ): Unit = {
    // Create the HiveContext and set it up for RDD->DF conversion
    val hc: HiveContext = new HiveContext(sc)

    // Convert the RDD into a schema'd DataFrame
    val df = hc.createDataFrame(rowsToSave).toDF(columnNames:_*)

    // Set up the hive context to support dynamic Hive partitions
    hc.setConf("hive.exec.dynamic.partition", "true")
    hc.setConf("hive.exec.dynamic.partition.mode", "nonstrict")

    // Drop the existing partition and write out the
    // DataFrame to Hive in a partition-aware manner
    dropHivePartition(hc, hiveTableName, hivePartitionSpecs)
    df.write
      .partitionBy(hivePartitionSpecs.keys.toSeq:_*)
      .mode(SaveMode.Append)
      .saveAsTable(hiveTableName)
  }

  /*
   * Given an RDD of Tuple (or case class) type rows,
   * save it to a partitioned Hive table.
   * The columns must be in the same order as the Tuple elements
   * (or the fields in the case class).
   */
  def saveDataframeToManagedPartitionedHiveTable(
      sc: SparkContext,
      df: DataFrame,
      hiveTableName: String,
      hivePartitionSpecs: Map[String, Any]
  ): Unit = {

    // Create the HiveContext and set it up for RDD->DF conversion
    val hc: HiveContext = new HiveContext(sc)

    // Set up the hive context to support dynamic Hive partitions
    hc.setConf("hive.exec.dynamic.partition", "true")
    hc.setConf("hive.exec.dynamic.partition.mode", "nonstrict")

    // Drop the existing partition and write out the
    // DataFrame to Hive in a partition-aware manner
    dropHivePartition(hc, hiveTableName, hivePartitionSpecs)
    df.write
      .partitionBy(hivePartitionSpecs.keys.toSeq: _*)
      .mode(SaveMode.Append)
      .saveAsTable(hiveTableName)
  }

  /*
   * Drop the specified Hive partition if it exists
   */
  def dropHivePartition(
      hc: HiveContext,
      hiveTableName: String,
      hivePartitionSpecs: Map[String, Any]
  ): Unit = {
    if (!hiveTableName.contains('.')) {
      throw new RuntimeException(s"Missing namespace for hive table: $hiveTableName.")
    }

    // Break the table name into namespace.table_name
    val Array(namespace, table) = hiveTableName.split('.')
    // Turn the partition spec map into a Hive-format String
    val partitionSpec = hivePartitionSpecsMapToString(hivePartitionSpecs)

    // Drop the partition if it exists
    hc.sql(s"USE $namespace")
    hc.sql(s"ALTER TABLE $table DROP IF EXISTS PARTITION ($partitionSpec)")
  }

  /*
   * Convert a partition_specs map to a usable Hive string.
   * ex:
   *   Map("ds" -> "2016-11-12", "night" -> 3)
   *   =>
   *   "ds='2016-11-12', night=3"
   */
  private def hivePartitionSpecsMapToString(hivePartitionSpecs: Map[String, Any]): String = {
    hivePartitionSpecs
      .map {
        // If the value is string type, surround it by single-quotes
        case (column, value: String) => s"$column='$value'"
        case (column, value) => s"$column=$value"
      }
      .mkString(", ")
  }

  def getLastPartition(hc: HiveContext, table: String): String = {
    val query = "SHOW PARTITIONS %s".format(table)
    val partitions = hc.sql(query).collect.map(_.getString(0))
    val lastPartition = partitions.max // ds=2015-01-01

    lastPartition.split("=").apply(1)
  }

  // ts convert ThriftDate to timestamp/1000/86400, e.g. number of days since 1970
  def ts(year: Int, month: Int, day: Int) : Long = {
    val date = new GregorianCalendar(year, month - 1, day)
    (date.getTimeInMillis/1000)/86400
  }

  def ts(ds: String): Long  = {
    ts(ds.substring(0,4).toInt, ds.substring(5,7).toInt, ds.substring(8,10).toInt)
  }

  // Returns a date iterator from start to end date.
  def dateRange(from: String, to: String): Seq[DateTime] = {
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val startDate = formatter.parseDateTime(from)
    val endDate = formatter.parseDateTime(to)
    val numberOfDays = Days.daysBetween(startDate, endDate).getDays

    for(n <- 0 to numberOfDays) yield startDate.plusDays(n)
  }

  // Returns a date iterator from today - from to today = to
  def dateRangeFromToday(from : Int, to : Int): Seq[DateTime] = {
    val now = new DateTime()
    val startDate = now.plusDays(from)
    val endDate = now.plusDays(to)
    val numberOfDays = Days.daysBetween(startDate, endDate).getDays

    for (n <- 0 to numberOfDays) yield startDate.plusDays(n)
  }

  def dateDiff(from: String, to: String): Int = {
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val startDate = formatter.parseDateTime(from)
    val endDate = formatter.parseDateTime(to)
    val numberOfDays = Days.daysBetween(startDate, endDate).getDays
    numberOfDays
  }

  def dateMinus(date: String, days: Int): String = {
    // return a string for the date which is 'days' earlier than 'date'
    // e.g. dateMinus("2015-06-01", 1) returns "2015-05-31"
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val dateFmt = formatter.parseDateTime(date)
    formatter.print(dateFmt.minusDays(days))
  }

  def datePlus(date: String, days: Int): String = {
    // return a string for the date which is 'days' earlier than 'date'
    // e.g. datePlus("2015-06-01", 1) returns "2015-06-02"
    dateMinus(date, -days)
  }

  // Returns a date range starting numDays before and ending at the desired date
  def dateRangeUntil(to: String, numDays: Int): Seq[DateTime] = {
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val endDate = formatter.parseDateTime(to)

    for(n <- numDays to 0 by -1) yield endDate.minusDays(n)
  }

  def dayOfYear(date: String): Int = {
    // return day of year of a date
    // e.g. dayOfYear("2015-01-01") returns 1
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val dateFmt = formatter.parseDateTime(date)
    dateFmt.dayOfYear().get()
  }

  def dayOfWeek(date: String): Int = {
    // return day of week of a date
    // e.g. if date is Monday, returns 1,
    // if date is Sunday, returns 0 (this is to be consistent with the pricing training data)
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val dateFmt = formatter.parseDateTime(date)
    dateFmt.dayOfWeek().get() % 7
  }

  def deleteHDFSFiles(path: String) = {
    logger.info("Deleting files at " + path)
    val hdfsPath = new Path(path)
    val fs = hdfsPath.getFileSystem(new Configuration())
    fs.delete(hdfsPath, true)
    logger.info("Deleted.")
  }

  def rowIsNotNull(row : Row, count : Int) : Boolean = {
    for (i <- 0 until count) {
      if (row.isNullAt(i)) return false
    }
    true
  }

  // Create a pair of spark accumulators that track (success, failure) counts.
  def addStatusAccumulators(sc: SparkContext,
      accName: String,
      accumulators: mutable.Map[String, Accumulator[Int]]): Unit = {
    accumulators.put(accName + ".success", sc.accumulator(0, accName + ".success"))
    accumulators.put(accName + ".failure", sc.accumulator(0, accName + ".failure"))
  }


  def countAllFailureCounters(accumulators: mutable.Map[String, Accumulator[Int]]): Long = {
    var failureCount = 0
    for ( (name, accumulator) <- accumulators) {
      logger.info("- Accumulator {} : {}", name, accumulator.value.toString )
      if (name.endsWith(".failure")) {
        failureCount += accumulator.value
      }
    }
    failureCount
  }

  def validateSuccessCounters(accumulators: mutable.Map[String, Accumulator[Int]],
      minSuccess: Int): Boolean = {
    for ( (name, accumulator) <- accumulators) {
      if (name.endsWith(".success") && accumulator.value < minSuccess) {
        logger.error("Failed counter: {} = {} < {}", name, accumulator.value.toString, minSuccess.toString)
        return false
      }
    }
    true
  }

  // TODO(kim): cleanup, write to hdfs directly instead of via SparkContext.
  def saveCountersAsTextFile(accumulators: mutable.Map[String, Accumulator[Int]],
      sc: SparkContext,
      hdfsFilePath: String): Unit = {
    var summary = Array("Summarizing counters:")

    for ( (name, accumulator) <- accumulators) {
      val logLine = "- %s = %d".format(name, accumulator.value )
      summary :+= logLine
      logger.info(logLine)
    }

    saveAndCommitAsTextFile(sc.parallelize(summary), hdfsFilePath, 1, true)
  }

  def getHDFSBufferedOutputStream(output: String): BufferedOutputStream = {
    new BufferedOutputStream(getHDFSOutputStream(output))
  }

  def getHDFSOutputStream(output: String): OutputStream = {
    val fs = FileSystem.get(new URI(output), hadoopConfiguration)
    val path = new Path(output)
    fs.create(path, true)
  }

  def getHDFSInputStreams(path: String): Array[(String, InputStream)] = {
    val fs = FileSystem.get(new Configuration())
    val status = fs.listStatus(new Path(path))
    status.map(f => (f.getPath.getName, fs.open(f.getPath)))
  }

  def getHDFSInputStream(input: String): InputStream = {
    val fs = FileSystem.get(new URI(input), hadoopConfiguration)
    val path = new Path(input)
    if (!fs.exists(path)) {
      logger.error(input + " does not exist")
      System.exit(-1)
    }
    fs.open(path)
  }

  def getHDFSBufferedReader(input: String): BufferedReader = {
    val inputStream = getHDFSInputStream(input)
    val reader = new BufferedReader(new InputStreamReader(inputStream))
    reader
  }

  def writeStringToFile(str: String, output: String) = {
    val stream = getHDFSOutputStream(output)
    val writer = new BufferedWriter(new OutputStreamWriter(stream))
    writer.write(str)
    writer.close()
  }

  def readStringFromFile(inputFile : String): String = {
    val stream = getHDFSInputStream(inputFile)
    val reader = new BufferedReader(new InputStreamReader(stream))
    val str = Stream.continually(reader.readLine()).takeWhile(_ != null).mkString("\n")
    str
  }

  def averageByKey[U:ClassTag](input: RDD[(U, Double)]) : RDD[(U, Double)] = {
    input
      .mapValues(value => (value, 1.0))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
      .mapValues(x => x._1 / x._2)
  }

  def copyFiles(srcPath: String, destPath: String, deleteSource: Boolean = false) = {
    val src = new Path(srcPath)
    val dest = new Path(destPath)
    val fsConfig = new Configuration()
    val fs = FileSystem.get(new java.net.URI(destPath), fsConfig)
    logger.info("Copying successful from " + src + " to " + dest)
    try {
      FileUtil.copy(fs, src, fs, dest, deleteSource, fsConfig)
    } catch {
      case e: Exception =>
        logger.info("Copy failed " + dest)
        System.exit(-1)
    }
    logger.info("Copy done.")
  }

  def getAllDirOrFiles(dirPath: String) : Array[String] = {
    // get all files under a directory
    val fsConfig = new Configuration()
    val fs = FileSystem.get(new java.net.URI(dirPath), fsConfig)
    val allFiles = fs.listStatus(new Path(dirPath))
    val out = new ArrayBuffer[String]()
    allFiles.foreach(f =>
      out.append(f.getPath.getName)
    )
    out.toArray
  }

  def countAnyNonEmpty(input: Any): Int = {
    // Return 0 if the input is empty, otherwise return 1
    // Note: this only supports checking for null, Optional.absent() and empty map
    if (input == null || Set("{}", "Optional.absent()", "()").contains(input.toString)) {
      0
    } else {
      1
    }
  }
}
