package com.airbnb.common.ml.search

import scala.util.Random

import com.airbnb.common.ml.util.{RandomUtil, ScalaLogging}


trait MonteCarloSearch {

  // return something the small the better.
  def eval(params: Map[String, Any]): Double

  // dispose called after each action,
  // for trainer has native resource, do clean up in dispose
  def dispose(): Unit

  // use List in this case to keep the order.
  def toString(params: List[(String, Any)]): String = {
    params.map(x => {
      x._2
    }).mkString("\001")
  }
}

object MonteCarloSearch extends ScalaLogging {

  def run(
      model: MonteCarloSearch,
      dynamicParams: List[(String, List[Any])],
      stableParams: List[(String, Any)],
      numIters: Int,
      tol: Double,
      minLoss: Double
  ): (String, Double) = {
    val randomizer = scala.util.Random
    var iters = 0
    var prevLoss = Double.MaxValue
    var bestParams = ""

    while (iters < numIters) {
      val currentParams = getParams(dynamicParams, randomizer)
      val finalParams = currentParams ::: stableParams
      val loss = model.eval(finalParams.toMap)
      logger.info(s" prevLoss $prevLoss, loss $loss")
      if (loss < prevLoss) {
        // keep the model, its obviously better than the previous one
        bestParams = model.toString(currentParams)
        // TODO save best model
        prevLoss = loss
      }
      iters += 1
      if (iters > 1) {
        val diff = loss - prevLoss
        if (diff < tol && loss < minLoss) {
          logger.info(s"search stop by diff $diff loss $loss")
          iters = numIters
        }
      }
    }
    model.dispose()

    // save bestParams.
    logger.info(s" bestParams $bestParams $prevLoss")
    (bestParams, prevLoss)
  }

  def getParams(
      params: List[(String, List[Any])],
      randomizer: Random
  ): List[(String, Any)] = {
    params.map((x) => {
      val choices = x._2
      if(choices.length == 2) {
        // pick random number between choices(0) and choices(1)
        (x._1, RandomUtil.randomNumber(choices, randomizer))
      } else if (choices.length == 1) {
        (x._1, choices.head)
      } else {
        // pick random index from choices
        (x._1, RandomUtil.randomIndex(choices, randomizer))
      }
    })
  }
}
