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
  if (args.find("silent") != args.cend()) {
    // Punch through, otherwise when silent == True is set this message will
    // never get displayed.
    LOG(CONSOLE)
        << "Parameter `silent` is deprecated, please use `verbosity` instead.";
  }

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
  switch (cur_verbosity_) {
    case LogVerbosity::kWarning:
      BaseLogger::log_stream_ << "WARNING: ";
    case LogVerbosity::kDebug:
      BaseLogger::log_stream_ << "DEBUG: ";
    case LogVerbosity::kInfo:
      BaseLogger::log_stream_ << "INFO: ";
    case LogVerbosity::kIgnore:
      BaseLogger::log_stream_ << file << ":" << line << ": ";
      break;
    case LogVerbosity::kSilent:
      break;
  }
}

}  // namespace xgboost
