package com.airbnb.common.ml.strategy.testutil

import java.io.InputStream

import scala.reflect.ClassTag

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object CSVUtil {
  def readCSVToLines(path: String): Iterator[Array[String]] = {
    val stream : InputStream = getClass.getResourceAsStream(path)
    val lines = scala.io.Source.fromInputStream( stream ).getLines.drop(1)
    lines.map( line =>
      line.split(",").map(_.stripPrefix("\"").stripSuffix("\"").trim)
    )
  }

  def parseCSVToSeq[T:ClassTag](name: String,
                                parseKey: (Array[String]) => String,
                                parseSample:(Array[String]) => T): Seq[(String, Seq[T])] = {
    val lines = readCSVToLines(name)
    val samples = lines.map(cols =>
      (parseKey(cols), parseSample(cols))
    ).toSeq.
      groupBy(_._1).map{
      case (key, seq) =>
        (key, seq.map(_._2))
    }.toSeq
    samples
  }

  def parseCSVToRDD[T:ClassTag](name: String,
                                parseKey: (Array[String]) => String,
                                parseSample:(Array[String]) => T,
                                sc: SparkContext): RDD[(String, Seq[T])] = {
    val samples = parseCSVToSeq(name, parseKey, parseSample)
    val rdd = sc.parallelize(samples)
    rdd
  }

}
