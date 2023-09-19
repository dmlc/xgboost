/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>                // for ASSERT_TRUE, ASSERT_EQ
#include <xgboost/collective/socket.h>  // for TCPSocket, Connect, SocketFinalize, SocketStartup
#include <xgboost/string_view.h>        // for StringView

#include <chrono>        // for seconds
#include <cstdint>       // for int8_t
#include <memory>        // for make_shared, shared_ptr
#include <system_error>  // for make_error_code, errc
#include <utility>       // for pair
#include <vector>        // for vector

#include "../../../src/collective/loop.h"  // for Loop

namespace xgboost::collective {
namespace {
class LoopTest : public ::testing::Test {
 protected:
  std::pair<TCPSocket, TCPSocket> pair_;
  std::shared_ptr<Loop> loop_;

 protected:
  void SetUp() override {
    system::SocketStartup();
    std::chrono::seconds timeout{1};

    auto domain = SockDomain::kV4;
    pair_.first = TCPSocket::Create(domain);
    auto port = pair_.first.BindHost();
    pair_.first.Listen();

    auto const& addr = SockAddrV4::Loopback().Addr();
    auto rc = Connect(StringView{addr}, port, 1, timeout, &pair_.second);
    ASSERT_TRUE(rc.OK());
    rc = pair_.second.NonBlocking(true);
    ASSERT_TRUE(rc.OK());

    pair_.first = pair_.first.Accept();
    rc = pair_.first.NonBlocking(true);
    ASSERT_TRUE(rc.OK());

    loop_ = std::make_shared<Loop>(timeout);
  }

  void TearDown() override {
    pair_ = decltype(pair_){};
    system::SocketFinalize();
  }
};
}  // namespace

TEST_F(LoopTest, Timeout) {
  std::vector<std::int8_t> data(1);
  Loop::Op op{Loop::Op::kRead, 0, data.data(), data.size(), &pair_.second, 0};
  loop_->Submit(op);
  auto rc = loop_->Block();
  ASSERT_FALSE(rc.OK());
  ASSERT_EQ(rc.Code(), std::make_error_code(std::errc::timed_out)) << rc.Report();
}

TEST_F(LoopTest, Op) {
  TCPSocket& send = pair_.first;
  TCPSocket& recv = pair_.second;

  std::vector<std::int8_t> wbuf(1, 1);
  std::vector<std::int8_t> rbuf(1, 0);

  Loop::Op wop{Loop::Op::kWrite, 0, wbuf.data(), wbuf.size(), &send, 0};
  Loop::Op rop{Loop::Op::kRead, 0, rbuf.data(), rbuf.size(), &recv, 0};

  loop_->Submit(wop);
  loop_->Submit(rop);

  auto rc = loop_->Block();
  ASSERT_TRUE(rc.OK()) << rc.Report();

  ASSERT_EQ(rbuf[0], wbuf[0]);
}
}  // namespace xgboost::collective
