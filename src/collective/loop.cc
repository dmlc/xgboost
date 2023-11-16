/**
 * Copyright 2023, XGBoost Contributors
 */
#include "loop.h"

#include <queue>  // for queue

#include "rabit/internal/socket.h"      // for PollHelper
#include "xgboost/collective/socket.h"  // for FailWithCode
#include "xgboost/logging.h"            // for CHECK

namespace xgboost::collective {
Result Loop::EmptyQueue(std::queue<Op>* p_queue) const {
  timer_.Start(__func__);
  auto error = [this] { timer_.Stop(__func__); };

  if (stop_) {
    timer_.Stop(__func__);
    return Success();
  }

  auto& qcopy = *p_queue;

  // clear the copied queue
  while (!qcopy.empty()) {
    rabit::utils::PollHelper poll;
    std::size_t n_ops = qcopy.size();

    // Iterate through all the ops for poll
    for (std::size_t i = 0; i < n_ops; ++i) {
      auto op = qcopy.front();
      qcopy.pop();

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

    // Iterate through all the ops for performing the operations
    for (std::size_t i = 0; i < n_ops; ++i) {
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
        auto rc = system::FailWithCode("Invalid socket output.");
        error();
        return rc;
      }

      op.off += n_bytes_done;
      CHECK_LE(op.off, op.n);

      if (op.off != op.n) {
        // not yet finished, push back to queue for next round.
        qcopy.push(op);
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

    auto unlock_notify = [&](bool is_blocking, bool stop) {
      if (!is_blocking) {
        std::lock_guard guard{mu_};
        stop_ = stop;
      } else {
        stop_ = stop;
        lock.unlock();
      }
      cv_.notify_one();
    };

    // move the queue
    std::queue<Op> qcopy;
    bool is_blocking = false;
    while (!queue_.empty()) {
      auto op = queue_.front();
      queue_.pop();
      if (op.code == Op::kBlock) {
        is_blocking = true;
      } else {
        qcopy.push(op);
      }
    }
    // unblock the queue
    if (!is_blocking) {
      lock.unlock();
    }
    // clear the queue
    auto rc = this->EmptyQueue(&qcopy);
    // Handle error
    if (!rc.OK()) {
      unlock_notify(is_blocking, true);
      std::lock_guard<std::mutex> guard{rc_lock_};
      this->rc_ = std::move(rc);
      return;
    }

    CHECK(qcopy.empty());
    unlock_notify(is_blocking, false);
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

[[nodiscard]] Result Loop::Block() {
  {
    std::lock_guard<std::mutex> guard{rc_lock_};
    if (!rc_.OK()) {
      return std::move(rc_);
    }
  }
  this->Submit(Op{Op::kBlock});
  {
    std::unique_lock lock{mu_};
    cv_.wait(lock, [this] { return (this->queue_.empty()) || stop_; });
  }
  {
    std::lock_guard<std::mutex> lock{rc_lock_};
    return std::move(rc_);
  }
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
