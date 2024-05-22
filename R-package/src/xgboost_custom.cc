// Copyright (c) 2015 by Contributors
// This file contains the customization implementations of R module
// to change behavior of libxgboost

#include <xgboost/logging.h>
#include "../../src/common/random.h"
#include "./xgboost_R.h"

// redirect the messages to R's console.
namespace dmlc {
void CustomLogMessage::Log(const std::string& msg) {
  Rprintf("%s\n", msg.c_str());
}
}  // namespace dmlc

namespace xgboost {
ConsoleLogger::~ConsoleLogger() {
  if (cur_verbosity_ == LogVerbosity::kIgnore ||
      cur_verbosity_ <= GlobalVerbosity()) {
    if (cur_verbosity_ == LogVerbosity::kWarning) {
      REprintf("%s\n", log_stream_.str().c_str());
    } else {
      dmlc::CustomLogMessage::Log(log_stream_.str());
    }
  }
}
TrackerLogger::~TrackerLogger() {
  dmlc::CustomLogMessage::Log(log_stream_.str());
}
}  // namespace xgboost

namespace xgboost {
namespace common {

// redirect the nath functions.
bool CheckNAN(double v) {
  return ISNAN(v);
}
#if !defined(XGBOOST_USE_CUDA)
double LogGamma(double v) {
  return lgammafn(v);
}
#endif  // !defined(XGBOOST_USE_CUDA)

}  // namespace common
}  // namespace xgboost
