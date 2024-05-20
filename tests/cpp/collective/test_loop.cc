/**
 * Copyright 2023-2024, XGBoost Contributors
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
    std::int32_t port{0};
    auto rc = Success() << [&] {
      return pair_.first.BindHost(&port);
    } << [&] {
      return pair_.first.Listen();
    };
    SafeColl(rc);

    auto const& addr = SockAddrV4::Loopback().Addr();
    rc = Connect(StringView{addr}, port, 1, timeout, &pair_.second);
    SafeColl(rc);
    rc = pair_.second.NonBlocking(true);
    SafeColl(rc);

    pair_.first = pair_.first.Accept();
    rc = pair_.first.NonBlocking(true);
    SafeColl(rc);

    loop_ = std::shared_ptr<Loop>{new Loop{timeout}};
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
  loop_->Submit(std::move(op));
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

  loop_->Submit(std::move(wop));
  loop_->Submit(std::move(rop));

  auto rc = loop_->Block();
  SafeColl(rc);

  ASSERT_EQ(rbuf[0], wbuf[0]);
}

TEST_F(LoopTest, Block) {
  // We need to ensure that a blocking call doesn't go unanswered.
  auto op = Loop::Op::Sleep(2);

  common::Timer t;
  t.Start();
  loop_->Submit(std::move(op));
  t.Stop();
  // submit is non-blocking
  ASSERT_LT(t.ElapsedSeconds(), 1);

  t.Start();
  auto rc = loop_->Block();
  t.Stop();
  SafeColl(rc);
  ASSERT_GE(t.ElapsedSeconds(), 1);
}
}  // namespace xgboost::collective
