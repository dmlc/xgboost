/**
 * Copyright 2018-2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/logging.h>

#include <map>

#include "capture_std.h"  // for CaptureStderr

namespace xgboost {
TEST(Logging, Basic) {
  auto verbosity = GlobalConfigThreadLocalStore::Get()->verbosity;
  std::map<std::string, std::string> args{};

  args["verbosity"] = "0";  // silent
  ConsoleLogger::Configure({args.cbegin(), args.cend()});
  auto output = CaptureStderr([] { LOG(DEBUG) << "Test silent."; });
  ASSERT_EQ(output.length(), 0);

  args["verbosity"] = "3";  // debug

  ConsoleLogger::Configure({args.cbegin(), args.cend()});
  output = CaptureStderr([&] { LOG(WARNING) << "Test Log Warning."; });
  ASSERT_NE(output.find("WARNING"), std::string::npos);

  output = CaptureStderr([&] { LOG(INFO) << "Test Log Info."; });
  ASSERT_NE(output.find("Test Log Info"), std::string::npos);

  output = CaptureStderr([&] { LOG(DEBUG) << "Test Log Debug."; });
  ASSERT_NE(output.find("DEBUG"), std::string::npos);

  args["verbosity"] = "1";  // warning
  ConsoleLogger::Configure({args.cbegin(), args.cend()});
  output = CaptureStderr([&] { LOG(INFO) << "INFO should not be displayed when set to warning."; });
  ASSERT_EQ(output.size(), 0);

  output = CaptureStderr([&] {
    LOG(CONSOLE) << "Test Log Console";  // ignore global setting.
  });
  ASSERT_NE(output.find("Test Log Console"), std::string::npos);

  args["verbosity"] = std::to_string(verbosity);
  ConsoleLogger::Configure({args.cbegin(), args.cend()});
}
}  // namespace xgboost
