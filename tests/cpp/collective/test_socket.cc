/**
 * Copyright 2022-2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/collective/socket.h>

#include <cerrno>        // EADDRNOTAVAIL
#include <system_error>  // std::error_code, std::system_category

#include "test_worker.h"  // for SocketTest

namespace xgboost::collective {
TEST_F(SocketTest, Basic) {
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

  if (SkipTest()) {
    GTEST_SKIP_(skip_msg_.c_str());
  }
  run_test(SockDomain::kV6);
}

TEST_F(SocketTest, Bind) {
  auto run = [](SockDomain domain) {
    auto any =
        domain == SockDomain::kV4 ? SockAddrV4::InaddrAny().Addr() : SockAddrV6::InaddrAny().Addr();
    auto sock = TCPSocket::Create(domain);
    std::int32_t port{0};
    auto rc = sock.Bind(any, &port);
    ASSERT_TRUE(rc.OK());
    ASSERT_NE(port, 0);
  };

  run(SockDomain::kV4);
  if (SkipTest()) {
    GTEST_SKIP_(skip_msg_.c_str());
  }
  run(SockDomain::kV6);
}
}  // namespace xgboost::collective
