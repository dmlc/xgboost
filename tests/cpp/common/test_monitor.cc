#include <gtest/gtest.h>
#include <xgboost/logging.h>
#include <string>
#include "../../../src/common/timer.h"

namespace xgboost {
namespace common {
TEST(Monitor, Basic) {
  auto run_monitor =
      []() {
        Monitor monitor_;
        monitor_.Init("Monitor test");
        monitor_.Start("basic");
        monitor_.Stop("basic");
      };

  ConsoleLogger::Configure({std::make_pair("verbosity", "3")});
  testing::internal::CaptureStderr();
  run_monitor();
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_NE(output.find("Monitor"), std::string::npos);

  ConsoleLogger::Configure({std::make_pair("verbosity", "2")});
  testing::internal::CaptureStderr();
  run_monitor();
  output = testing::internal::GetCapturedStderr();
  ASSERT_EQ(output.find("Monitor"), std::string::npos);
}
}  // namespace common
}  // namespace xgboost
