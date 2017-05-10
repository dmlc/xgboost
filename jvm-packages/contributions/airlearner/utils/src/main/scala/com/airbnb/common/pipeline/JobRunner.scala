package com.airbnb.common.pipeline

import scala.tools.scalap.scalax.rules.Zero
import scala.util.{Failure, Success, Try}

import com.typesafe.config.Config
import org.apache.spark.{SparkConf, SparkContext}

import com.airbnb.common.config.AirCon
import com.airbnb.common.ml.util.ScalaLogging


trait JobRunner extends ScalaLogging {
  def getRegisterKryoClasses: List[Class[_]] = {
    List()
  }

  // customize setup for JobRunner
  def setup(sparkConf: SparkConf, appName: String, jobName: String): Unit

  // customize jobs.
  def makeJob(sparkContext: SparkContext, jobName: String, configs: Array[Config]): Job

  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      logger.error("Usage: JobRunner jobName configName1,configName2...")
      System.exit(-1)
    }

    val jobName = args(0)
    val configs = args.slice(1, args.length).map(AirCon.load)

    val sparkConf = new SparkConf()

    val appName = if (System.getProperty("app_name") != null) {
      // Support custom app names for debugging purposes
      System.getProperty("app_name")
    } else {
      jobName
    }

    sparkConf.setAppName(appName)

    // http://spark.apache.org/docs/latest/tuning.html
    // Set Kryo as default serializer and register common classes to
    // optimize serialization performance
    val default = List(
      classOf[Array[Float]],
      classOf[Array[Int]],
      classOf[Array[Array[Float]]],
      classOf[Zero],
      classOf[java.util.HashMap[_, _]],
      classOf[(_, _)]
    )
    sparkConf.registerKryoClasses(
      (default ::: getRegisterKryoClasses).toArray
    )

    setup(sparkConf, appName, jobName)

    val sparkContext = new SparkContext(sparkConf)

    val job = makeJob(sparkContext, jobName, configs)

    val result = Try(job.run())

    sparkContext.stop()

    result match {
      case Success(_) =>
        logger.info(s"Job $jobName succeeded.")
        System.exit(0)
      case Failure(e) =>
        logger.info(s"Job $jobName failed.")
        e.printStackTrace()
        System.exit(-1)
    }
  }
}
