/**
 * Copyright 2023, XGBoost Contributors
 */
#include "loop.h"

#include <queue>  // for queue

#include "rabit/internal/socket.h"      // for PollHelper
#include "xgboost/collective/socket.h"  // for FailWithCode
#include "xgboost/logging.h"            // for CHECK

namespace xgboost::collective {
Result Loop::EmptyQueue() {
  timer_.Start(__func__);
  auto error = [this] {
    this->stop_ = true;
    timer_.Stop(__func__);
  };

  while (!queue_.empty() && !stop_) {
    std::queue<Op> qcopy;
    rabit::utils::PollHelper poll;

    // watch all ops
    while (!queue_.empty()) {
      auto op = queue_.front();
      queue_.pop();

      switch (op.code) {
        case Op::kRead: {
          poll.WatchRead(*op.sock);
          break;
        }
        case Op::kWrite: {
          poll.WatchWrite(*op.sock);
          break;
        }
        default: {
          error();
          return Fail("Invalid socket operation.");
        }
      }
      qcopy.push(op);
    }

    // poll, work on fds that are ready.
    timer_.Start("poll");
    auto rc = poll.Poll(timeout_);
    timer_.Stop("poll");
    if (!rc.OK()) {
      error();
      return rc;
    }
    // we wonldn't be here if the queue is empty.
    CHECK(!qcopy.empty());

    while (!qcopy.empty() && !stop_) {
      auto op = qcopy.front();
      qcopy.pop();

      std::int32_t n_bytes_done{0};
      CHECK(op.sock->NonBlocking());

      switch (op.code) {
        case Op::kRead: {
          if (poll.CheckRead(*op.sock)) {
            n_bytes_done = op.sock->Recv(op.ptr + op.off, op.n - op.off);
          }
          break;
        }
        case Op::kWrite: {
          if (poll.CheckWrite(*op.sock)) {
            n_bytes_done = op.sock->Send(op.ptr + op.off, op.n - op.off);
          }
          break;
        }
        default: {
          error();
          return Fail("Invalid socket operation.");
        }
      }

      if (n_bytes_done == -1 && !system::LastErrorWouldBlock()) {
        stop_ = true;
        auto rc = system::FailWithCode("Invalid socket output.");
        error();
        return rc;
      }
      op.off += n_bytes_done;
      CHECK_LE(op.off, op.n);

      if (op.off != op.n) {
        // not yet finished, push back to queue for next round.
        queue_.push(op);
      }
    }
  }
  timer_.Stop(__func__);
  return Success();
}

void Loop::Process() {
  // consumer
  while (true) {
    std::unique_lock lock{mu_};
    cv_.wait(lock, [this] { return !this->queue_.empty() || stop_; });
    if (stop_) {
      break;
    }
    CHECK(!mu_.try_lock());

    this->rc_ = this->EmptyQueue();
    if (!rc_.OK()) {
      stop_ = true;
      cv_.notify_one();
      break;
    }

    CHECK(queue_.empty());
    CHECK(!mu_.try_lock());
    cv_.notify_one();
  }

  if (rc_.OK()) {
    CHECK(queue_.empty());
  }
}

Result Loop::Stop() {
  std::unique_lock lock{mu_};
  stop_ = true;
  lock.unlock();

  CHECK_EQ(this->Block().OK(), this->rc_.OK());

  if (curr_exce_) {
    std::rethrow_exception(curr_exce_);
  }

  return Success();
}

Loop::Loop(std::chrono::seconds timeout) : timeout_{timeout} {
  timer_.Init(__func__);
  worker_ = std::thread{[this] {
    try {
      this->Process();
    } catch (std::exception const& e) {
      std::lock_guard<std::mutex> guard{mu_};
      if (!curr_exce_) {
        curr_exce_ = std::current_exception();
        rc_ = Fail("Exception was thrown");
      }
      stop_ = true;
      cv_.notify_all();
    } catch (...) {
      std::lock_guard<std::mutex> guard{mu_};
      if (!curr_exce_) {
        curr_exce_ = std::current_exception();
        rc_ = Fail("Exception was thrown");
      }
      stop_ = true;
      cv_.notify_all();
    }
  }};
}
}  // namespace xgboost::collective
