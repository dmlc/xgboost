/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include "loop.h"

#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t
#include <exception>  // for exception, current_exception, rethrow_exception
#include <future>     // for promise
#include <memory>     // for make_shared
#include <mutex>      // for lock_guard, unique_lock
#include <queue>      // for queue
#include <string>     // for string
#include <thread>     // for thread
#include <utility>    // for move

#include "../common/threading_utils.h"      // for NameThread
#include "xgboost/collective/poll_utils.h"  // for PollHelper
#include "xgboost/collective/result.h"      // for Fail, Success
#include "xgboost/collective/socket.h"      // for FailWithCode
#include "xgboost/logging.h"                // for CHECK

namespace xgboost::collective {
Result Loop::ProcessQueue(std::queue<Op>* p_queue) const {
  timer_.Start(__func__);
  auto error = [this](Op op) {
    op.pr->set_value();
    timer_.Stop(__func__);
  };

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
      auto op = std::move(qcopy.front());
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
        case Op::kSleep: {
          break;
        }
        default: {
          error(op);
          return Fail("Invalid socket operation.");
        }
      }

      qcopy.push(std::move(op));
    }

    // poll, work on fds that are ready.
    timer_.Start("poll");
    if (!poll.fds.empty()) {
      auto rc = poll.Poll(timeout_);
      if (!rc.OK()) {
        timer_.Stop(__func__);
        return rc;
      }
    }
    timer_.Stop("poll");

    // We wonldn't be here if the queue is empty.
    CHECK(!qcopy.empty());

    // Iterate through all the ops for performing the operations
    for (std::size_t i = 0; i < n_ops; ++i) {
      auto op = std::move(qcopy.front());
      qcopy.pop();

      std::int32_t n_bytes_done{0};
      if (!op.sock) {
        CHECK(op.code == Op::kSleep);
      } else {
        CHECK(op.sock->NonBlocking());
      }

      switch (op.code) {
        case Op::kRead: {
          if (poll.CheckRead(*op.sock)) {
            n_bytes_done = op.sock->Recv(op.ptr + op.off, op.n - op.off);
            if (n_bytes_done == 0) {
              error(op);
              return Fail("Encountered EOF. The other end is likely closed.",
                          op.sock->GetSockError());
            }
          }
          break;
        }
        case Op::kWrite: {
          if (poll.CheckWrite(*op.sock)) {
            n_bytes_done = op.sock->Send(op.ptr + op.off, op.n - op.off);
          }
          break;
        }
        case Op::kSleep: {
          // For testing only.
          std::this_thread::sleep_for(std::chrono::seconds{op.n});
          n_bytes_done = op.n;
          break;
        }
        default: {
          error(op);
          return Fail("Invalid socket operation.");
        }
      }

      if (n_bytes_done == -1 && !system::LastErrorWouldBlock()) {
        auto rc = system::FailWithCode("Invalid socket output.");
        error(op);
        return rc;
      }

      op.off += n_bytes_done;
      CHECK_LE(op.off, op.n);

      if (op.off != op.n) {
        // not yet finished, push back to queue for the next round.
        qcopy.push(op);
      } else {
        op.pr->set_value();
      }
    }
  }

  timer_.Stop(__func__);
  return Success();
}

void Loop::Process() {
  auto set_rc = [this](Result&& rc) {
    std::lock_guard lock{rc_lock_};
    rc_ = std::forward<Result>(rc);
  };

  // This loop cannot exit unless `stop_` is set to true. There must always be a thread to
  // answer the call even if there are errors.
  while (true) {
    try {
      std::unique_lock lock{mu_};
      // This can handle missed notification: wait(lock, predicate) is equivalent to:
      //
      // while (!predicate()) {
      //    cv.wait(lock);
      // }
      //
      // As a result, if there's a missed notification, the queue wouldn't be empty, hence
      // the predicate would be false and the actual wait wouldn't be invoked. Therefore,
      // the blocking call can never go unanswered.
      cv_.wait(lock, [this] { return !this->queue_.empty() || stop_; });
      if (stop_) {
        break;  // only point where this loop can exit.
      }

      // Move the global queue into a local variable to unblock it.
      std::queue<Op> qcopy;

      while (!queue_.empty()) {
        auto op = std::move(queue_.front());
        queue_.pop();
        qcopy.push(op);
      }
      lock.unlock();

      // Clear the local queue.
      auto rc = this->ProcessQueue(&qcopy);

      // Handle error
      if (!rc.OK()) {
        set_rc(std::move(rc));
      } else {
        std::unique_lock lock{mu_};
        CHECK(qcopy.empty() || stop_);
      }
    } catch (std::exception const& e) {
      curr_exce_ = std::current_exception();
      set_rc(Fail("Exception inside the event loop:" + std::string{e.what()}));
    } catch (...) {
      curr_exce_ = std::current_exception();
      set_rc(Fail("Unknown exception inside the event loop."));
    }
  }
}

Result Loop::Stop() {
  // Finish all remaining tasks
  CHECK_EQ(this->Block().OK(), this->rc_.OK());

  // Notify the loop to stop
  std::unique_lock lock{mu_};
  stop_ = true;
  lock.unlock();
  this->cv_.notify_one();

  if (this->worker_.joinable()) {
    this->worker_.join();
  }

  if (curr_exce_) {
    std::rethrow_exception(curr_exce_);
  }

  return Success();
}

[[nodiscard]] Result Loop::Block() {
  {
    // Check whether the last op was successful, stop if not.
    std::lock_guard<std::mutex> guard{rc_lock_};
    if (!rc_.OK()) {
      stop_ = true;
    }
  }
  if (!this->worker_.joinable()) {
    std::lock_guard<std::mutex> guard{rc_lock_};
    return Fail("Worker has stopped.", std::move(rc_));
  }

  {
    std::unique_lock lock{mu_};
    cv_.notify_one();
  }

  for (auto& fut : futures_) {
    if (fut.valid()) {
      try {
        fut.get();
      } catch (std::future_error const&) {
        // Do nothing. If something went wrong in the worker, we have a std::future_error
        // due to broken promise. This function will transfer the rc back to the caller.
      }
    }
  }
  futures_.clear();

  {
    // Transfer the rc.
    std::lock_guard<std::mutex> lock{rc_lock_};
    return std::move(rc_);
  }
}

void Loop::Submit(Op op) {
  auto p = std::make_shared<std::promise<void>>();
  op.pr = std::move(p);
  futures_.emplace_back(op.pr->get_future());
  CHECK_NE(op.n, 0);

  std::unique_lock lock{mu_};
  queue_.push(op);
}

Loop::Loop(std::chrono::seconds timeout) : timeout_{timeout} {
  timer_.Init(__func__);
  worker_ = std::thread{[this] {
    this->Process();
  }};
  common::NameThread(&worker_, "lw");
}
}  // namespace xgboost::collective
