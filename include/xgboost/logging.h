/*!
 * Copyright (c) 2015 by Contributors
 * \file logging.h
 * \brief defines console logging options for xgboost.
 *  Use to enforce unified print behavior.
 *  For debug loggers, use LOG(INFO) and LOG(ERROR).
 */
#ifndef XGBOOST_LOGGING_H_
#define XGBOOST_LOGGING_H_

#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <sstream>
#include "./base.h"

namespace xgboost {

class BaseLogger {
 public:
  BaseLogger() {
#if XGBOOST_LOG_WITH_TIME
    log_stream_ << "[" << dmlc::DateLogger().HumanDate() << "] ";
#endif
  }
  std::ostream& stream() { return log_stream_; }  // NOLINT

 protected:
  std::ostringstream log_stream_;
};

class ConsoleLogger : public BaseLogger {
 public:
  ~ConsoleLogger();
};

class TrackerLogger : public BaseLogger {
 public:
  ~TrackerLogger();
};

class LogCallbackRegistry {
 public:
  using Callback = void (*)(const char*);
  LogCallbackRegistry()
    : log_callback_([] (const char* msg) { std::cerr << msg << std::endl; }) {}
  inline void Register(Callback log_callback) {
    this->log_callback_ = log_callback;
  }
  inline Callback Get() const {
    return log_callback_;
  }
 private:
  Callback log_callback_;
};

using LogCallbackRegistryStore = dmlc::ThreadLocalStore<LogCallbackRegistry>;

// redefines the logging macro if not existed
#ifndef LOG
#define LOG(severity) LOG_##severity.stream()
#endif

// Enable LOG(CONSOLE) for print messages to console.
#define LOG_CONSOLE ::xgboost::ConsoleLogger()
// Enable LOG(TRACKER) for print messages to tracker
#define LOG_TRACKER ::xgboost::TrackerLogger()
}  // namespace xgboost.
#endif  // XGBOOST_LOGGING_H_
