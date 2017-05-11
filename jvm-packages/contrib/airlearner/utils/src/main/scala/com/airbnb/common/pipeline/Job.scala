package com.airbnb.common.pipeline

import com.typesafe.config.Config
import org.apache.spark.SparkContext

import com.airbnb.common.ml.util.ScalaLogging


object Job extends ScalaLogging {

  /**
    * Convert an entry-point in a pricing pipeline to a PricingJob.
    */
  def singleConfigJobRunner(
      sc: SparkContext,
      configs: Array[Config],
      runnerFunction: (SparkContext, Config) => Unit): Job = {
    assert(configs.length == 1, "Wrong number of configs provided")

    new Job {
      override def run(): Unit = {
        val config: Config = configs(0)
        logger.info(s"Running job with config: ${config.root().render()}")
        runnerFunction(sc, config)
      }
    }
  }

  def singleConfigJobRunner(
      sc: SparkContext,
      configs: Array[Config],
      configPath: String,
      runnerFunction: (SparkContext, Config) => Unit): Job = {
    assert(configs.length == 1, "Wrong number of configs provided")

    new Job {
      override def run(): Unit = {
        val config: Config = configs(0).getConfig(configPath)
        logger.info(s"Running job with $configPath config: ${config.root().render()}")
        runnerFunction(sc, config)
      }
    }
  }

  def singleConfigJobRunner(
      sc: SparkContext,
      configs: Array[Config],
      configPath1: String,
      configPath2: String,
      runnerFunction: (SparkContext, Config, Config) => Unit): Job = {
    assert(configs.length == 1, "Wrong number of configs provided")

    new Job {
      override def run(): Unit = {
        val config1: Config = configs(0).getConfig(configPath1)
        logger.info(s"Running job with $configPath1 config: ${config1.root().render()}")
        val config2: Config = configs(0).getConfig(configPath2)
        logger.info(s"Running job with $configPath2 config: ${config2.root().render()}")

        runnerFunction(sc, config1, config2)
      }
    }
  }
}

trait Job {
  def run(): Unit
}
