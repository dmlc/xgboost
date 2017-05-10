package com.airbnb.common.ml.util

import scala.util.Random


object RandomUtil {
  def randomDouble(bounds: List[Any], randomizer: Random): Double = {
    val min = bounds.head.asInstanceOf[Double]
    val max = bounds.tail.head.asInstanceOf[Double]
    (randomizer.nextDouble * (max - min)) + min
  }

  def randomInt(bounds: List[Any], randomizer: Random): Int = {
    val min = bounds.head.asInstanceOf[Int]
    val max = bounds.tail.head.asInstanceOf[Int]
    randomizer.nextInt(max - min) + min
  }

  def randomNumber(bounds: List[Any], randomizer: Random): Any = {
    if (bounds.head.isInstanceOf[Int]) {
      randomInt(bounds, randomizer)
    } else {
      randomDouble(bounds, randomizer)
    }
  }

  def randomIndex(bounds: List[Any], randomizer: Random): Any = {
    val index = randomizer.nextInt(bounds.length)
    bounds(index)
  }

  def sample[T](list: Seq[T], ratios: List[Double]): List[Seq[T]]= {
    val t = Random.shuffle(list)
    slice(t, ratios)
  }

  def slice[T](list: Seq[T], ratios: List[Double]): List[Seq[T]]= {
    val len = list.length
    val start = ratios.scanLeft(0.0)(_ + _).take(ratios.length)

    start.zip(ratios).map{
      case (s:Double, size:Double) => {
        val startPos: Int = (s * len).toInt
        val endPos: Int = ((s + size)*len).toInt
        list.slice(startPos, endPos)
      }
    }
  }
}
