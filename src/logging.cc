/*!
 * Copyright 2015 by Contributors
 * \file logging.cc
 * \brief Implementation of loggers.
 * \author Tianqi Chen
 */
#include <xgboost/logging.h>
#include <iostream>
#include "./common/sync.h"

#if !defined(XGBOOST_STRICT_R_MODE) || XGBOOST_STRICT_R_MODE == 0
// Override logging mechanism for non-R interfaces
void dmlc::CustomLogMessage::Log(const std::string& msg) {
  const xgboost::LogCallbackRegistry* registry
    = xgboost::LogCallbackRegistryStore::Get();
  auto callback = registry->Get();
  callback(msg.c_str());
}

namespace xgboost {
ConsoleLogger::~ConsoleLogger() {
  dmlc::CustomLogMessage::Log(log_stream_.str());
}

TrackerLogger::~TrackerLogger() {
  log_stream_ << '\n';
  rabit::TrackerPrint(log_stream_.str());
}

}  // namespace xgboost
#endif
