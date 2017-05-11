package com.airbnb.common.ml.strategy.data

import scala.util.Random


object BinarySampleMockUtil {
  private val MaxPrice: Double = 200.0

  def getKnownSample: BaseBinarySample = {
    BaseBinarySample(label = true, 0.7, 100, 90, Some(100))
  }

  def getKnownSamples: Seq[BaseBinarySample] = {
    Seq(
      BaseBinarySample(label = true, 0.7, 100, 100, Some(100)),
      BaseBinarySample(label = true, 0.6, 100, 100, Some(100)),
      BaseBinarySample(label = false, 0.5, 90, 95, None),
      BaseBinarySample(label = true, 0.3, 85, 100, Some(100))
    )
  }

  def getRandomSample: BaseBinarySample = {
    val label: Boolean = Random.nextBoolean()
    BaseBinarySample(
      label = label,
      Random.nextDouble(),
      Random.nextDouble() * MaxPrice,
      Random.nextDouble() * MaxPrice,
      if (label) Some(Random.nextDouble() * MaxPrice) else None
    )
  }

  def getRandomSamples(numSamples: Int): Seq[BaseBinarySample] = {
    Seq.fill(numSamples)(getRandomSample)
  }
}
