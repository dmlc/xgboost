package com.airbnb.common.ml.util

import scala.util.Random


object RandomUtil {
  def randomDouble(bounds: Seq[Any], randomizer: Random): Double = {
    val min = bounds.head.asInstanceOf[Double]
    val max = bounds.tail.head.asInstanceOf[Double]
    (randomizer.nextDouble * (max - min)) + min
  }

  def randomInt(bounds: Seq[Any], randomizer: Random): Int = {
    val min = bounds.head.asInstanceOf[Int]
    val max = bounds.tail.head.asInstanceOf[Int]
    randomizer.nextInt(max - min) + min
  }

  def randomNumber(bounds: Seq[Any], randomizer: Random): Any = {
    if (bounds.head.isInstanceOf[Int]) {
      randomInt(bounds, randomizer)
    } else {
      randomDouble(bounds, randomizer)
    }
  }

  def randomIndex(bounds: Seq[Any], randomizer: Random): Any = {
    val index = randomizer.nextInt(bounds.length)
    bounds(index)
  }

  def sample[T](items: Seq[T], ratios: Seq[Double]): Seq[Seq[T]]= {
    val t = Random.shuffle(items)
    slice(t, ratios)
  }

  def slice[T](items: Seq[T], ratios: Seq[Double]): Seq[Seq[T]]= {
    val len = items.length
    val start = ratios.scanLeft(0.0)(_ + _).take(ratios.length)

    start.zip(ratios).map{
      case (s:Double, size:Double) => {
        val startPos: Int = (s * len).toInt
        val endPos: Int = ((s + size)*len).toInt
        items.slice(startPos, endPos)
      }
    }
  }
}
