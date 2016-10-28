/*!
 * Copyright 2015 by Contributors
 * \file logging.cc
 * \brief Implementation of loggers.
 * \author Tianqi Chen
 */
#include <xgboost/logging.h>
#include <iostream>
#include "./common/sync.h"

namespace xgboost {

#if XGBOOST_CUSTOMIZE_LOGGER == 0
ConsoleLogger::~ConsoleLogger() {
  std::cerr << log_stream_.str() << std::endl;
}

TrackerLogger::~TrackerLogger() {
  log_stream_ << '\n';
  rabit::TrackerPrint(log_stream_.str());
}
#endif
}  // namespace xgboost
