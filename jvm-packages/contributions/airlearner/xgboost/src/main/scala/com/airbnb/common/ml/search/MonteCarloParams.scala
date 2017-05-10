package com.airbnb.common.ml.search

import scala.collection.JavaConverters._

import com.typesafe.config.Config


object MonteCarloParams {

  def loadFromConfig(config: Config): List[(String, List[Any])] = {
    val doubleParams: List[(String, List[Double])] = loadParams(config, "double_params", toDouble)
    val intParams: List[(String, List[Int])] = loadParams(config, "int_params", toInt)
    doubleParams ::: intParams
  }

  def loadParams[T](config: Config, path: String, convert: (Any) => T): List[(String, List[T])] = {
    if (config.hasPath(path)) {
      val names = config.getStringList(path).asScala.toList
      var paramArray = List[(String, List[T])]()
      for (param <- names) {
        val list = config.getAnyRefList(param).asScala.toList.map(convert)
        paramArray = paramArray.::((param, list))
      }
      // .:: append to head, so reverse it.
      paramArray.reverse
    } else {
      List.empty
    }
  }

  def toDouble: (Any) => Double = {
    case i: Int => i
    case f: Float => f
    case d: Double => d
  }

  def toInt: (Any) => Int = {
    case i: Int => i
  }

  def getRangeDouble(center: Double, range: List[Double], searchRatio: Double): List[Double] = {
    val distance = range(1) - range.head
    val searchDistance = searchRatio * distance
    val right = math.min(searchDistance + center, range(1))
    val left = math.max(center - searchDistance, range.head)
    List(left, right)
  }

  def getIntRange(center: Int, range: List[Int], searchRatio: Double): List[Int] = {
    val distance = range(1) - range.head
    val searchDistance = searchRatio * distance
    if (searchDistance <= 1) {
      List(center)
    } else {
      val right = math.min(searchDistance + center, range(1)).toInt
      val left = math.max(center - searchDistance, range.head).toInt
      List(left, right)
    }
  }

  def getRange(center: Double, range: List[Any], searchRatio: Double): List[Any] = {
    range.head match {
      case head: Double => getRangeDouble(center, range.asInstanceOf[List[Double]], searchRatio)
      case head: Int => getIntRange(center.toInt, range.asInstanceOf[List[Int]], searchRatio)
    }
  }

  // order of param and lastParam should be same
  def adjustParamsFromPreviousSearch(
      param: List[(String, List[Any])],
      lastParam: List[Double],
      searchRatio: Double
  ): List[(String, List[Any])] = {
    assert(param.length == lastParam.length)
    param.zip(lastParam).map {
      case ((name, range), center) => {
        (name, getRange(center, range, searchRatio))
      }
    }
  }
}
