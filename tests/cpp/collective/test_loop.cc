/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <chrono>        // for seconds
#include <cinttypes>     // for int8_t
#include <memory>        // for make_shared
#include <system_error>  // make_error_code, errc

#include "../../../src/collective/loop.h"

namespace xgboost::collective {

class LoopTest : public ::testing::Test {
  void SetUp() override { system::SocketStartup(); }
  void TearDown() override { system::SocketFinalize(); }
};

TEST_F(LoopTest, Timeout) {
  std::chrono::seconds timeout{1};
  auto loop = std::make_shared<Loop>(timeout);

  TCPSocket sock;
  std::vector<std::int8_t> data(1);
  Loop::Op op{Loop::Op::kRead, 0, data.data(), data.size(), &sock, 0};
  loop->Submit(op);
  auto rc = loop->Block();
  ASSERT_FALSE(rc.OK());
  ASSERT_EQ(rc.Code(), std::make_error_code(std::errc::timed_out)) << rc.Report();
}

TEST_F(LoopTest, Op) {
  auto domain = SockDomain::kV4;
  auto server = TCPSocket::Create(domain);
  auto port = server.BindHost();
  server.Listen();

  TCPSocket send;
  auto const& addr = SockAddrV4::Loopback().Addr();
  auto rc = Connect(StringView{addr}, port, 1, std::chrono::seconds{3}, &send);
  ASSERT_TRUE(rc.OK()) << rc.Report();
  rc = send.NonBlocking(true);
  ASSERT_TRUE(rc.OK()) << rc.Report();

  auto recv = server.Accept();
  rc = recv.NonBlocking(true);
  ASSERT_TRUE(rc.OK()) << rc.Report();

  std::vector<std::int8_t> wbuf(1, 1);
  std::vector<std::int8_t> rbuf(1, 0);

  std::chrono::seconds timeout{1};
  auto loop = std::make_shared<Loop>(timeout);

  Loop::Op wop{Loop::Op::kWrite, 0, wbuf.data(), wbuf.size(), &send, 0};
  Loop::Op rop{Loop::Op::kRead, 0, rbuf.data(), rbuf.size(), &recv, 0};

  loop->Submit(wop);
  loop->Submit(rop);

  rc = loop->Block();
  ASSERT_TRUE(rc.OK()) << rc.Report();

  ASSERT_EQ(rbuf[0], wbuf[0]);
}
}  // namespace xgboost::collective
