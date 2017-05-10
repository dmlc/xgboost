package com.airbnb.common.ml.util

import java.io.{BufferedReader, IOException, InputStreamReader}
import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}


object HDFSUtil extends ScalaLogging {

  private lazy val hadoopConfiguration = new Configuration()

  /**
    * if path exists and no "/_temporary", last task succeed
    * else, last task failed.
    * remove "/_temporary" if it exists.
    * @param path String
    * @return true if last task succeed
    */
  def lastTaskSucceed(path: String): Boolean = {
    if (dirExists(path)) {
      if (dirExists(path + "/_temporary")) {
        logger.info(s"Deleting partial data for $path.")
        deleteDirWithoutThrow(path)
        false
      } else {
        logger.info(s"$path exists")
        true
      }
    } else {
      logger.info(s"$path does not exist")
      false
    }
  }

  def dirExists(dir: String): Boolean = {
    val path = new Path(dir)
    val hdfs = FileSystem.get(
      new java.net.URI(dir), hadoopConfiguration)

    hdfs.exists(path)
  }

  def deleteDirWithoutThrow(dir: String): Unit = {
    val path = new Path(dir)
    val hdfs = FileSystem.get(
      new java.net.URI(dir), hadoopConfiguration)
    if (hdfs.exists(path)) {
      logger.warn(s"$dir exists, DELETING")
      try {
        hdfs.delete(path, true)
      } catch {
        case e: IOException => logger.error(s" exception $e")
      }
    }
  }

  def createPath(path: String): Unit = {
    val remotePath = new Path(path)
    val remoteFS = remotePath.getFileSystem(hadoopConfiguration)
    remoteFS.mkdirs(new Path(path))
  }

  def readStringFromFile(inputFile : String): String = {
    val fs = FileSystem.get(new URI(inputFile), hadoopConfiguration)
    val path = new Path(inputFile)
    val stream = fs.open(path)
    val reader = new BufferedReader(new InputStreamReader(stream))
    val str = Stream.continually(reader.readLine()).takeWhile(_ != null).mkString("\n")
    str
  }

}
