package com.airbnb.common.ml.util

import scala.collection.JavaConverters._
import scala.collection.immutable.Map

import com.typesafe.config.Config


/**
  * Utils collection handling typesafe Config objects
  */
object ConfigUtils {

  /**
    * Given a Config object, convert it to a map of key(string) to objects.
    *
    * @param config the Config object
    * @return the converted map
    */
  def configToMap(config: Config): Map[String, Any] = {
    config.entrySet().asScala.map (
      entry => {
        val value = entry.getValue.unwrapped()
        val obj = value match {
          case l: java.util.List[_] => l.asInstanceOf[java.util.List[_]].asScala
          case _ => value
        }

        (entry.getKey, obj)
      }
    ).toMap
  }

  def getMapStringList(config: Config, path: String, delimiter: String): Map[String, String] = {
    config
      .getStringList(path)
      .asScala
      .map(line => line.split(delimiter))
      .filter(_.length == 2)
      .map(kv => kv(0) -> kv(1))
      .toMap
  }
}
