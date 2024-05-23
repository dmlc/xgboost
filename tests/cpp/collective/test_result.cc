/**
 *  Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/collective/result.h>

namespace xgboost::collective {
TEST(Result, Concat) {
  auto rc0 = Fail("foo");
  auto rc1 = Fail("bar");
  auto rc = std::move(rc0) + std::move(rc1);
  ASSERT_NE(rc.Report().find("foo"), std::string::npos);
  ASSERT_NE(rc.Report().find("bar"), std::string::npos);

  auto rc2 = Fail("Another", std::move(rc));
  auto assert_that = [](Result const& rc) {
    ASSERT_NE(rc.Report().find("Another"), std::string::npos);
    ASSERT_NE(rc.Report().find("foo"), std::string::npos);
    ASSERT_NE(rc.Report().find("bar"), std::string::npos);
  };
  assert_that(rc2);

  auto empty = Success();
  auto rc3 = std::move(empty) + std::move(rc2);
  assert_that(rc3);

  empty = Success();
  auto rc4 = std::move(rc3) + std::move(empty);
  assert_that(rc4);
}
}  // namespace xgboost::collective
