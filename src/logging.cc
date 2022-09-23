/*!
 * Copyright 2015-2018 by Contributors
 * \file logging.cc
 * \brief Implementation of loggers.
 * \author Tianqi Chen
 */
#include <iostream>

#include "xgboost/parameter.h"
#include "xgboost/logging.h"

#include "collective/communicator-inl.h"

#if !defined(XGBOOST_STRICT_R_MODE) || XGBOOST_STRICT_R_MODE == 0
// Override logging mechanism for non-R interfaces
void dmlc::CustomLogMessage::Log(const std::string& msg) {
  const xgboost::LogCallbackRegistry *registry =
      xgboost::LogCallbackRegistryStore::Get();
  auto callback = registry->Get();
  callback(msg.c_str());
}

namespace xgboost {

ConsoleLogger::~ConsoleLogger() {
  if (ShouldLog(cur_verbosity_)) {
    dmlc::CustomLogMessage::Log(BaseLogger::log_stream_.str());
  }
}

TrackerLogger::~TrackerLogger() {
  log_stream_ << '\n';
  collective::Print(log_stream_.str());
}

}  // namespace xgboost

#endif  // !defined(XGBOOST_STRICT_R_MODE) || XGBOOST_STRICT_R_MODE == 0

namespace xgboost {

bool ConsoleLogger::ShouldLog(LogVerbosity verbosity) {
  return static_cast<int>(verbosity) <=
             (GlobalConfigThreadLocalStore::Get()->verbosity) ||
         verbosity == LV::kIgnore;
}

void ConsoleLogger::Configure(Args const& args) {
  auto& param = *GlobalConfigThreadLocalStore::Get();
  param.UpdateAllowUnknown(args);
}

ConsoleLogger::LogVerbosity ConsoleLogger::DefaultVerbosity() {
  return LogVerbosity::kWarning;
}

ConsoleLogger::LogVerbosity ConsoleLogger::GlobalVerbosity() {
  LogVerbosity global_verbosity { LogVerbosity::kWarning };
  switch (GlobalConfigThreadLocalStore::Get()->verbosity) {
  case 0:
    global_verbosity = LogVerbosity::kSilent;
    break;
  case 1:
    global_verbosity = LogVerbosity::kWarning;
    break;
  case 2:
    global_verbosity = LogVerbosity::kInfo;
    break;
  case 3:
    global_verbosity = LogVerbosity::kDebug;
  default:
    // global verbosity doesn't require kIgnore
    break;
  }

  return global_verbosity;
}

ConsoleLogger::ConsoleLogger(LogVerbosity cur_verb) :
    cur_verbosity_{cur_verb} {}

ConsoleLogger::ConsoleLogger(
    const std::string& file, int line, LogVerbosity cur_verb) {
  cur_verbosity_ = cur_verb;
  switch (cur_verbosity_) {
    case LogVerbosity::kWarning:
      BaseLogger::log_stream_ << "WARNING: "
                              << file << ":" << line << ": ";
      break;
    case LogVerbosity::kDebug:
      BaseLogger::log_stream_ << "DEBUG: "
                              << file << ":" << line << ": ";
      break;
    case LogVerbosity::kInfo:
      BaseLogger::log_stream_ << "INFO: "
                              << file << ":" << line << ": ";
      break;
    case LogVerbosity::kIgnore:
      BaseLogger::log_stream_ << file << ":" << line << ": ";
      break;
    case LogVerbosity::kSilent:
      break;
  }
}

}  // namespace xgboost
