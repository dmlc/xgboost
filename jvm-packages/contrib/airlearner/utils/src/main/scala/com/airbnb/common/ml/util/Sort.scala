package com.airbnb.common.ml.util

import scala.annotation.tailrec
import scala.util.Random


object Sort {
  @tailrec
  def quickSelect[A](
      seq: Seq[A], n: Int, rand: Random = new Random)(implicit evidence: A => Ordered[A]): A = {
    assert(n < seq.length, s"n $n cannot be larger than length of sequence ${seq.length}")
    val pivot = rand.nextInt(seq.length)
    val (left, right) = seq.partition(seq(pivot).>)
    if (left.length == n) {
      seq(pivot)
    } else if (left.isEmpty) {
      val (left, right) = seq.partition(seq(pivot).==)
      if (left.length > n) {
        seq(pivot)
      } else {
        quickSelect(right, n - left.length, rand)
      }
    } else if (left.length < n) {
      quickSelect(right, n - left.length, rand)
    } else {
      quickSelect(left, n, rand)
    }
  }

  def quickSelectAxis[A](
      seq: Array[Array[A]], n: Int, axis: Int, rand: Random = new Random)(implicit evidence: A => Ordered[A]): A = {
    implicit val ordering = Ordering.by[Array[A], A](_(axis))
    val found = quickSelect(seq, n, rand)
    // need to be separate line due to implicit
    found(axis)
  }

}
