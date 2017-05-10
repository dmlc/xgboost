package com.airbnb.common.config;

import com.typesafe.config.Config;

public interface MacroExecutor {

  String getMaxString(Config config);
  String getMinString(Config config);
  String getSum(Config config);
  String getMatchedLatestDirectory(Config config);
  String getDateMinusDays(Config config);
  String getHdfsPartitionsByDates(Config config);
}
