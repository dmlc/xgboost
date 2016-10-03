/*!
 * Copyright (c) 2015 by Contributors
 * \file concurrency.h
 * \brief thread-safe data structures.
 * \author Yutian Li
 */
#ifndef DMLC_CONCURRENCY_H_
#define DMLC_CONCURRENCY_H_
// this code depends on c++11
#if DMLC_USE_CXX11
#include <atomic>
#include <queue>
#include <mutex>
#include <vector>
#include <condition_variable>
#include "dmlc/base.h"

namespace dmlc {

/*!
 * \brief Simple userspace spinlock implementation.
 */
class Spinlock {
 public:
#ifdef _MSC_VER
  Spinlock() {
    lock_.clear();
  }
#else
  Spinlock() : lock_(ATOMIC_FLAG_INIT) {
  }
#endif
  ~Spinlock() = default;
  /*!
   * \brief Acquire lock.
   */
  inline void lock() noexcept(true);
  /*!
   * \brief Release lock.
   */
  inline void unlock() noexcept(true);

 private:
  std::atomic_flag lock_;
  /*!
   * \brief Disable copy and move.
   */
  DISALLOW_COPY_AND_ASSIGN(Spinlock);
};

/*! \brief type of concurrent queue */
enum class ConcurrentQueueType {
  /*! \brief FIFO queue */
  kFIFO,
  /*! \brief queue with priority */
  kPriority
};

/*!
 * \brief Cocurrent blocking queue.
 */
template <typename T,
          ConcurrentQueueType type = ConcurrentQueueType::kFIFO>
class ConcurrentBlockingQueue {
 public:
  ConcurrentBlockingQueue();
  ~ConcurrentBlockingQueue() = default;
  /*!
   * \brief Push element into the queue.
   * \param e Element to push into.
   * \param priority the priority of the element, only used for priority queue.
   *            The higher the priority is, the better.
   * \tparam E the element type
   *
   * It will copy or move the element into the queue, depending on the type of
   * the parameter.
   */
  template <typename E>
  void Push(E&& e, int priority = 0);
  /*!
   * \brief Pop element from the queue.
   * \param rv Element popped.
   * \return On false, the queue is exiting.
   *
   * The element will be copied or moved into the object passed in.
   */
  bool Pop(T* rv);
  /*!
   * \brief Signal the queue for destruction.
   *
   * After calling this method, all blocking pop call to the queue will return
   * false.
   */
  void SignalForKill();
  /*!
   * \brief Get the size of the queue.
   * \return The size of the queue.
   */
  size_t Size();

 private:
  struct Entry {
    T data;
    int priority;
    inline bool operator<(const Entry &b) const {
      return priority < b.priority;
    }
  };

  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> exit_now_;
  int nwait_consumer_;
  // a priority queue
  std::vector<Entry> priority_queue_;
  // a FIFO queue
  std::queue<T> fifo_queue_;
  /*!
   * \brief Disable copy and move.
   */
  DISALLOW_COPY_AND_ASSIGN(ConcurrentBlockingQueue);
};

inline void Spinlock::lock() noexcept(true) {
  while (lock_.test_and_set(std::memory_order_acquire)) {
  }
}

inline void Spinlock::unlock() noexcept(true) {
  lock_.clear(std::memory_order_release);
}

template <typename T, ConcurrentQueueType type>
ConcurrentBlockingQueue<T, type>::ConcurrentBlockingQueue()
    : exit_now_{false}, nwait_consumer_{0} {}

template <typename T, ConcurrentQueueType type>
template <typename E>
void ConcurrentBlockingQueue<T, type>::Push(E&& e, int priority) {
  static_assert(std::is_same<typename std::remove_cv<
                                 typename std::remove_reference<E>::type>::type,
                             T>::value,
                "Types must match.");
  bool notify;
  {
    std::lock_guard<std::mutex> lock{mutex_};
    if (type == ConcurrentQueueType::kFIFO) {
      fifo_queue_.emplace(std::forward<E>(e));
      notify = nwait_consumer_ != 0;
    } else {
      Entry entry;
      entry.data = std::move(e);
      entry.priority = priority;
      priority_queue_.push_back(std::move(entry));
      std::push_heap(priority_queue_.begin(), priority_queue_.end());
      notify = nwait_consumer_ != 0;
    }
  }
  if (notify) cv_.notify_one();
}

template <typename T, ConcurrentQueueType type>
bool ConcurrentBlockingQueue<T, type>::Pop(T* rv) {
  std::unique_lock<std::mutex> lock{mutex_};
  if (type == ConcurrentQueueType::kFIFO) {
    ++nwait_consumer_;
    cv_.wait(lock, [this] {
        return !fifo_queue_.empty() || exit_now_.load();
      });
    --nwait_consumer_;
    if (!exit_now_.load()) {
      *rv = std::move(fifo_queue_.front());
      fifo_queue_.pop();
      return true;
    } else {
      return false;
    }
  } else {
    ++nwait_consumer_;
    cv_.wait(lock, [this] {
        return !priority_queue_.empty() || exit_now_.load();
      });
    --nwait_consumer_;
    if (!exit_now_.load()) {
      std::pop_heap(priority_queue_.begin(), priority_queue_.end());
      *rv = std::move(priority_queue_.back().data);
      priority_queue_.pop_back();
      return true;
    } else {
      return false;
    }
  }
}

template <typename T, ConcurrentQueueType type>
void ConcurrentBlockingQueue<T, type>::SignalForKill() {
  {
    std::lock_guard<std::mutex> lock{mutex_};
    exit_now_.store(true);
  }
  cv_.notify_all();
}

template <typename T, ConcurrentQueueType type>
size_t ConcurrentBlockingQueue<T, type>::Size() {
  std::lock_guard<std::mutex> lock{mutex_};
  if (type == ConcurrentQueueType::kFIFO) {
    return fifo_queue_.size();
  } else {
    return priority_queue_.size();
  }
}
}  // namespace dmlc
#endif  // DMLC_USE_CXX11
#endif  // DMLC_CONCURRENCY_H_
