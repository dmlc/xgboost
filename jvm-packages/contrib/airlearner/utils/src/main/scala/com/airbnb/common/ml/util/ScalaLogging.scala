package com.airbnb.common.ml.util

// scalastyle:off ban.logger.factory
import com.typesafe.scalalogging.slf4j.Logger
import org.slf4j.LoggerFactory


/**
  * This trait imbues it's inheriting class with a transient logger field.
  * This is to be used on case classes or other classes which extend Serializable.
  * The purpose is to avoid serializing and passing around the logger itself.
  * Passing around a logger (especially over the network in Spark) is wasteful
  * and unnecessary.
  *
  * Based upon `LazyLogging`.
  */
trait ScalaLogging {
  @transient
  protected lazy val logger: Logger = Logger(LoggerFactory.getLogger(getClass.getName))
}
// scalastyle:on ban.logger.factory
