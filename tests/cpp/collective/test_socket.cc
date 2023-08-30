/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/collective/socket.h>

#include <cerrno>        // EADDRNOTAVAIL
#include <fstream>       // ifstream
#include <system_error>  // std::error_code, std::system_category

#include "../helpers.h"

namespace xgboost::collective {
TEST(Socket, Basic) {
  system::SocketStartup();

  SockAddress addr{SockAddrV6::Loopback()};
  ASSERT_TRUE(addr.IsV6());
  addr = SockAddress{SockAddrV4::Loopback()};
  ASSERT_TRUE(addr.IsV4());

  std::string msg{"Skipping IPv6 test"};

  auto run_test = [msg](SockDomain domain) {
    auto server = TCPSocket::Create(domain);
    ASSERT_EQ(server.Domain(), domain);
    auto port = server.BindHost();
    server.Listen();

    TCPSocket client;
    if (domain == SockDomain::kV4) {
      auto const& addr = SockAddrV4::Loopback().Addr();
      auto rc = Connect(StringView{addr}, port, 1, std::chrono::seconds{3}, &client);
      ASSERT_TRUE(rc.OK()) << rc.Report();
    } else {
      auto const& addr = SockAddrV6::Loopback().Addr();
      auto rc = Connect(StringView{addr}, port, 1, std::chrono::seconds{3}, &client);
      // some environment (docker) has restricted network configuration.
      if (!rc.OK() && rc.Code() == std::error_code{EADDRNOTAVAIL, std::system_category()}) {
        GTEST_SKIP_(msg.c_str());
      }
      ASSERT_EQ(rc, Success()) << rc.Report();
    }
    ASSERT_EQ(client.Domain(), domain);

    auto accepted = server.Accept();
    StringView msg{"Hello world."};
    accepted.Send(msg);

    std::string str;
    client.Recv(&str);
    ASSERT_EQ(StringView{str}, msg);
  };

  run_test(SockDomain::kV4);

  std::string path{"/sys/module/ipv6/parameters/disable"};
  if (FileExists(path)) {
    std::ifstream fin(path);
    if (!fin) {
      GTEST_SKIP_(msg.c_str());
    }
    std::string s_value;
    fin >> s_value;
    auto value = std::stoi(s_value);
    if (value != 0) {
      GTEST_SKIP_(msg.c_str());
    }
  } else {
    GTEST_SKIP_(msg.c_str());
  }
  run_test(SockDomain::kV6);

  system::SocketFinalize();
}
}  // namespace xgboost::collective
