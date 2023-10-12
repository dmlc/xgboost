/**
 * Copyright 2022-2023, XGBoost Contributors
 */
#pragma once

#include <gtest/gtest.h>
#include <xgboost/collective/socket.h>

#include <fstream>  // ifstream

#include "../helpers.h"  // for FileExists

namespace xgboost::collective {
class SocketTest : public ::testing::Test {
 protected:
  std::string skip_msg_{"Skipping IPv6 test"};

  bool SkipTest() {
    std::string path{"/sys/module/ipv6/parameters/disable"};
    if (FileExists(path)) {
      std::ifstream fin(path);
      if (!fin) {
        return true;
      }
      std::string s_value;
      fin >> s_value;
      auto value = std::stoi(s_value);
      if (value != 0) {
        return true;
      }
    } else {
      return true;
    }
    return false;
  }

 protected:
  void SetUp() override { system::SocketStartup(); }
  void TearDown() override { system::SocketFinalize(); }
};
}  // namespace xgboost::collective
