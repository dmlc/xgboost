/*!
 * Copyright 2015-2018 by Contributors
 * \file logging.cc
 * \brief Implementation of loggers.
 * \author Tianqi Chen
 */
#include <rabit/rabit.h>
#include <dmlc/parameter.h>
#include <xgboost/logging.h>

#include <iostream>
#include <map>

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
  if (cur_verbosity_ == LogVerbosity::kIgnore ||
      cur_verbosity_ <= global_verbosity_) {
    dmlc::CustomLogMessage::Log(BaseLogger::log_stream_.str());
  }
}

TrackerLogger::~TrackerLogger() {
  log_stream_ << '\n';
  rabit::TrackerPrint(log_stream_.str());
}

}  // namespace xgboost

#endif

namespace xgboost {

DMLC_REGISTER_PARAMETER(ConsoleLoggerParam);

ConsoleLogger::LogVerbosity ConsoleLogger::global_verbosity_ =
    ConsoleLogger::DefaultVerbosity();

ConsoleLoggerParam ConsoleLogger::param_ = ConsoleLoggerParam();
void ConsoleLogger::Configure(const std::map<std::string, std::string>& args) {
  param_.InitAllowUnknown(args);
  // Deprecated, but when trying to display deprecation message some R
  // tests trying to catch stdout will fail.
  if (param_.silent) {
    global_verbosity_ = LogVerbosity::kSilent;
    return;
  }
  switch (param_.verbosity) {
    case 0:
      global_verbosity_ = LogVerbosity::kSilent;
      break;
    case 1:
      global_verbosity_ = LogVerbosity::kWarning;
      break;
    case 2:
      global_verbosity_ = LogVerbosity::kInfo;
      break;
    case 3:
      global_verbosity_ = LogVerbosity::kDebug;
    default:
      // global verbosity doesn't require kIgnore
      break;
  }
}

ConsoleLogger::LogVerbosity ConsoleLogger::DefaultVerbosity() {
  return LogVerbosity::kWarning;
}

ConsoleLogger::LogVerbosity ConsoleLogger::GlobalVerbosity() {
  return global_verbosity_;
}

ConsoleLogger::ConsoleLogger() : cur_verbosity_{LogVerbosity::kInfo} {}
ConsoleLogger::ConsoleLogger(LogVerbosity cur_verb) :
    cur_verbosity_{cur_verb} {}

ConsoleLogger::ConsoleLogger(
    const std::string& file, int line, LogVerbosity cur_verb) {
  cur_verbosity_ = cur_verb;
  auto add_msg =
      [&, this](std::string const& msg="") {
        this->BaseLogger::log_stream_ << msg
                                      << file << ":" << line << ": ";
      };
  switch (cur_verbosity_) {
    case LogVerbosity::kWarning:
      add_msg("WARNING: ");
      break;
    case LogVerbosity::kDebug:
      add_msg("DEBUG: ");
      break;
    case LogVerbosity::kInfo:
      add_msg("INFO: ");
      break;
    case LogVerbosity::kIgnore:
      add_msg();
      break;
    case LogVerbosity::kSilent:
      break;
  }
}

}  // namespace xgboost
