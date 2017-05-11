package com.airbnb.common.ml.util

import org.joda.time.format.DateTimeFormat
import org.joda.time.{DateTime, DateTimeConstants}


object DateTimeUtil {

  private val DATE_FORMATTER = DateTimeFormat.forPattern("yyyy-MM-dd")

  def getDateTimeFromString(dateStr: String): DateTime = {
    DateTime.parse(dateStr, DATE_FORMATTER)
  }

  def getDaysSinceEpoch(date: DateTime): Long = {
    date.getMillis / DateTimeConstants.MILLIS_PER_DAY
  }

  def dateStringFromInt(daysSinceEpoch: Long): String = {
    DATE_FORMATTER.print(daysSinceEpoch * DateTimeConstants.MILLIS_PER_DAY)
  }

  def daysRange(begin: String, end: String): Int = {
    (
      getDaysSinceEpoch(getDateTimeFromString(end)) -
        getDaysSinceEpoch(getDateTimeFromString(begin))
      ).toInt
  }
}
