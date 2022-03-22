/*!
 * Copyright (c) 2015-2019 by Contributors
 * \file logging.h
 *
 * \brief defines console logging options for xgboost.  Use to enforce unified print
 *  behavior.
 */
#ifndef XGBOOST_LOGGING_H_
#define XGBOOST_LOGGING_H_

#include <dmlc/logging.h>
#include <dmlc/thread_local.h>

#include <xgboost/base.h>
#include <xgboost/parameter.h>
#include <xgboost/global_config.h>

#include <sstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace xgboost {

class BaseLogger {
 public:
  BaseLogger() {
#if XGBOOST_LOG_WITH_TIME
    log_stream_ << "[" << dmlc::DateLogger().HumanDate() << "] ";
#endif  // XGBOOST_LOG_WITH_TIME
  }
  std::ostream& stream() { return log_stream_; }  // NOLINT

 protected:
  std::ostringstream log_stream_;
};

class ConsoleLogger : public BaseLogger {
 public:
  enum class LogVerbosity {
    kSilent = 0,
    kWarning = 1,
    kInfo = 2,   // information may interests users.
    kDebug = 3,  // information only interesting to developers.
    kIgnore = 4  // ignore global setting
  };
  using LV = LogVerbosity;

 private:
  LogVerbosity cur_verbosity_;

 public:
  static void Configure(Args const& args);

  static LogVerbosity GlobalVerbosity();
  static LogVerbosity DefaultVerbosity();
  static bool ShouldLog(LogVerbosity verbosity);

  ConsoleLogger() = delete;
  explicit ConsoleLogger(LogVerbosity cur_verb);
  ConsoleLogger(const std::string& file, int line, LogVerbosity cur_verb);
  ~ConsoleLogger();
};

class TrackerLogger : public BaseLogger {
 public:
  ~TrackerLogger();
};

// custom logging callback; disabled for R wrapper
#if !defined(XGBOOST_STRICT_R_MODE) || XGBOOST_STRICT_R_MODE == 0
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
#else
class LogCallbackRegistry {
 public:
  using Callback = void (*)(const char*);
  LogCallbackRegistry() {}
  inline void Register(Callback log_callback) {}
  inline Callback Get() const {
    return nullptr;
  }
};
#endif  // !defined(XGBOOST_STRICT_R_MODE) || XGBOOST_STRICT_R_MODE == 0

using LogCallbackRegistryStore = dmlc::ThreadLocalStore<LogCallbackRegistry>;

// Redefines LOG_WARNING for controling verbosity
#if defined(LOG_WARNING)
#undef  LOG_WARNING
#endif  // defined(LOG_WARNING)
#define LOG_WARNING                                                            \
  if (::xgboost::ConsoleLogger::ShouldLog(                                     \
          ::xgboost::ConsoleLogger::LV::kWarning))                             \
  ::xgboost::ConsoleLogger(__FILE__, __LINE__,                                 \
                           ::xgboost::ConsoleLogger::LogVerbosity::kWarning)

// Redefines LOG_INFO for controling verbosity
#if defined(LOG_INFO)
#undef  LOG_INFO
#endif  // defined(LOG_INFO)
#define LOG_INFO                                                               \
  if (::xgboost::ConsoleLogger::ShouldLog(                                     \
          ::xgboost::ConsoleLogger::LV::kInfo))                                \
  ::xgboost::ConsoleLogger(__FILE__, __LINE__,                                 \
                           ::xgboost::ConsoleLogger::LogVerbosity::kInfo)

#if defined(LOG_DEBUG)
#undef LOG_DEBUG
#endif  // defined(LOG_DEBUG)
#define LOG_DEBUG                                                              \
  if (::xgboost::ConsoleLogger::ShouldLog(                                     \
          ::xgboost::ConsoleLogger::LV::kDebug))                               \
  ::xgboost::ConsoleLogger(__FILE__, __LINE__,                                 \
                           ::xgboost::ConsoleLogger::LogVerbosity::kDebug)

// redefines the logging macro if not existed
#ifndef LOG
#define LOG(severity) LOG_##severity.stream()
#endif  // LOG

// Enable LOG(CONSOLE) for print messages to console.
#define LOG_CONSOLE ::xgboost::ConsoleLogger(           \
    ::xgboost::ConsoleLogger::LogVerbosity::kIgnore)
// Enable LOG(TRACKER) for print messages to tracker
#define LOG_TRACKER ::xgboost::TrackerLogger()

#if defined(CHECK)
#undef CHECK
#define CHECK(cond)                                     \
  if (XGBOOST_EXPECT(!(cond), false))                   \
    dmlc::LogMessageFatal(__FILE__, __LINE__).stream()  \
        << "Check failed: " #cond << ": "
#endif  // defined(CHECK)

}  // namespace xgboost.
#endif  // XGBOOST_LOGGING_H_
