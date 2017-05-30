package com.airbnb.common.ml.strategy.config

import com.typesafe.config.Config
import scala.collection.JavaConverters._
import scala.util.Try

/**
  * @param paramNames parameter names
  * @param paramCombinations each row consists of one set of parameters
  */
case class SearchParams[T](paramNames: List[String], paramCombinations: List[List[T]])

object SearchParams {
  val DOUBLE_PARAMS = "params"
  val LIST_DOUBLE_PARAMS = "list_params"

  private def loadFromConfig[T](config: Config,
                                configName: String,
                                parse: (Config, String) => List[T]): SearchParams[T] = {
    val paramNames = config.getStringList(configName).asScala.toList
    var paramArray = List[List[T]]()
    for (param <- paramNames) {
      paramArray = paramArray :+ parse(config, param)
    }
    val paramCombinations = generateParamsCombination(paramArray)
    SearchParams(paramNames, paramCombinations)
  }

  def prettyPrint[T](paramNames: List[String], paramValues: List[T]): String = {
    require(paramNames.length == paramValues.length)
    paramNames
      .zip(paramValues)
      .map(x => x._1 + "=" + x._2.toString)
      .mkString(",")
  }

  def getSearchParam(config: Config): Either[SearchParams[Double], SearchParams[List[Double]]] = {
    val useList = Try(config.getBoolean("use_list_params")).getOrElse(false)

    if (useList) {
      Right(loadListDoubleFromConfig(config))
    } else {
      Left(loadDoubleFromConfig(config))
    }
  }

  def loadDoubleFromConfig(config: Config): SearchParams[Double] = {
    loadFromConfig(config, DOUBLE_PARAMS, parseDouble)
  }

  def loadListDoubleFromConfig(config: Config): SearchParams[List[Double]] = {
    loadFromConfig(config, LIST_DOUBLE_PARAMS, parseListDouble)
  }

  def parseDouble(config: Config, param: String): List[Double] = {
    config.getDoubleList(param).asScala.toList.map(_.toDouble)
  }

  // min: [{list:[0.0, 1, -4]}, {list:[0.1, 2, -5]}]
  // must use list as the indicator string
  def parseListDouble(config: Config, param: String): List[List[Double]] = {
    config.getConfigList(param).asScala.toList.map(p => {
      p.getDoubleList("list").asScala.toList.map(_.toDouble)
    }
    )
  }

  def getParamNameAndValueFromPrettyPrint(in: String): (List[String], List[String]) = {
    if (in.isEmpty || !in.contains("="))
      return (List(""), List(""))

    val arr = in.split(",").map(x => x.split('='))
    (arr.map(x => x(0)).toList, arr.map(x => x(1)).toList)
  }

  private def generateParamsCombination[T](x: List[List[T]]): List[List[T]] = x match {
    case Nil => List(Nil)
    case h :: _ => h.flatMap(i => generateParamsCombination(x.tail).map(i :: _))
  }
}
