/**
 * Copyright 2023, XGBoost Contributors
 */
#include "../../../src/collective/tracker.h"  // for GetHostAddress
#include "net_test.h"                         // for SocketTest

namespace xgboost::collective {
namespace {
class TrackerTest : public SocketTest {};
}  // namespace

TEST_F(TrackerTest, GetHostAddress) {
  std::string host;
  auto rc = GetHostAddress(&host);
  ASSERT_TRUE(rc.OK());
  ASSERT_TRUE(host.find("127.") == std::string::npos);
}
}  // namespace xgboost::collective
