#include <gtest/gtest.h>
#include <xgboost/logging.h>
#include <string>
#include "../../../src/common/timer.h"

namespace xgboost {
namespace common {
TEST(Monitor, Logging) {
  auto run_monitor =
      []() {
        Monitor monitor_;
        monitor_.Init("Monitor test");
        monitor_.Start("basic");
        monitor_.Stop("basic");
      };

  Args args = {std::make_pair("verbosity", "3")};
  ConsoleLogger::Configure(args);
  ASSERT_EQ(ConsoleLogger::GlobalVerbosity(), ConsoleLogger::LogVerbosity::kDebug);

  testing::internal::CaptureStderr();
  run_monitor();
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_NE(output.find("Monitor"), std::string::npos);

  // Monitor only prints messages when set to DEBUG.
  args = {std::make_pair("verbosity", "2")};
  ConsoleLogger::Configure(args);
  testing::internal::CaptureStderr();
  run_monitor();
  output = testing::internal::GetCapturedStderr();
  ASSERT_EQ(output.size(), 0);

  ConsoleLogger::Configure(Args{{"verbosity", "1"}});
}
}  // namespace common
}  // namespace xgboost
