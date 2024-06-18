/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include "loop.h"

#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t
#include <exception>  // for exception, current_exception, rethrow_exception
#include <mutex>      // for lock_guard, unique_lock
#include <queue>      // for queue
#include <string>     // for string
#include <thread>     // for thread
#include <utility>    // for move

#include "rabit/internal/socket.h"      // for PollHelper
#include "xgboost/collective/result.h"  // for Fail, Success
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
  auto set_rc = [this](Result&& rc) {
    std::lock_guard lock{rc_lock_};
    rc_ = std::forward<Result>(rc);
  };

  // This loop cannot exit unless `stop_` is set to true. There must always be a thread to
  // answer the blocking call even if there are errors, otherwise the blocking will wait
  // forever.
  while (true) {
    try {
      std::unique_lock lock{mu_};
      cv_.wait(lock, [this] { return !this->queue_.empty() || stop_; });
      if (stop_) {
        break;  // only point where this loop can exit.
      }

      // Move the global queue into a local variable to unblock it.
      std::queue<Op> qcopy;

      bool is_blocking = false;
      while (!queue_.empty()) {
        auto op = queue_.front();
        queue_.pop();
        if (op.code == Op::kBlock) {
          is_blocking = true;
          // Block must be the last op in the current batch since no further submit can be
          // issued until the blocking call is finished.
          CHECK(queue_.empty());
        } else {
          qcopy.push(op);
        }
      }

      if (!is_blocking) {
        // Unblock, we can write to the global queue again.
        lock.unlock();
      }

      // Clear the local queue, this is blocking the current worker thread (but not the
      // client thread), wait until all operations are finished.
      auto rc = this->EmptyQueue(&qcopy);

      if (is_blocking) {
        // The unlock is delayed if this is a blocking call
        lock.unlock();
      }

      // Notify the client thread who called block after all error conditions are set.
      auto notify_if_block = [&] {
        if (is_blocking) {
          std::unique_lock lock{mu_};
          block_done_ = true;
          lock.unlock();
          block_cv_.notify_one();
        }
      };

      // Handle error
      if (!rc.OK()) {
        set_rc(std::move(rc));
      } else {
        CHECK(qcopy.empty());
      }

      notify_if_block();
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

  this->Submit(Op{Op::kBlock});

  {
    // Wait for the block call to finish.
    std::unique_lock lock{mu_};
    block_cv_.wait(lock, [this] { return block_done_ || stop_; });
    block_done_ = false;
  }

  {
    // Transfer the rc.
    std::lock_guard<std::mutex> lock{rc_lock_};
    return std::move(rc_);
  }
}

Loop::Loop(std::chrono::seconds timeout) : timeout_{timeout} {
  timer_.Init(__func__);
  worker_ = std::thread{[this] { this->Process(); }};
}
}  // namespace xgboost::collective
