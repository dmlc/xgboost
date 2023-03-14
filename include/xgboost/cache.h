/**
 * Copyright 2023 by XGBoost contributors
 */
#ifndef XGBOOST_CACHE_H_
#define XGBOOST_CACHE_H_

#include <xgboost/logging.h>  // for CHECK_EQ, CHECK

#include <cstddef>            // for size_t
#include <memory>             // for weak_ptr, shared_ptr, make_shared
#include <mutex>              // for mutex, lock_guard
#include <queue>              // for queue
#include <thread>             // for thread
#include <unordered_map>      // for unordered_map
#include <utility>            // for move
#include <vector>             // for vector

namespace xgboost {
class DMatrix;
/**
 * \brief Thread-aware FIFO cache for DMatrix related data.
 *
 * \tparam CacheT The type that needs to be cached.
 */
template <typename CacheT>
class DMatrixCache {
 public:
  struct Item {
    // A weak pointer for checking whether the DMatrix object has expired.
    std::weak_ptr<DMatrix> ref;
    // The cached item
    std::shared_ptr<CacheT> value;

    CacheT const& Value() const { return *value; }
    CacheT& Value() { return *value; }

    Item(std::shared_ptr<DMatrix> m, std::shared_ptr<CacheT> v) : ref{m}, value{std::move(v)} {}
  };

  static constexpr std::size_t DefaultSize() { return 32; }

 private:
  mutable std::mutex lock_;

 protected:
  struct Key {
    DMatrix const* ptr;
    std::thread::id const thread_id;

    bool operator==(Key const& that) const {
      return ptr == that.ptr && thread_id == that.thread_id;
    }
  };
  struct Hash {
    std::size_t operator()(Key const& key) const noexcept {
      std::size_t f = std::hash<DMatrix const*>()(key.ptr);
      std::size_t s = std::hash<std::thread::id>()(key.thread_id);
      if (f == s) {
        return f;
      }
      return f ^ s;
    }
  };

  std::unordered_map<Key, Item, Hash> container_;
  std::queue<Key> queue_;
  std::size_t max_size_;

  void CheckConsistent() const { CHECK_EQ(queue_.size(), container_.size()); }

  void ClearExpired() {
    // Clear expired entries
    this->CheckConsistent();
    std::vector<Key> expired;
    std::queue<Key> remained;

    while (!queue_.empty()) {
      auto p_fmat = queue_.front();
      auto it = container_.find(p_fmat);
      CHECK(it != container_.cend());
      if (it->second.ref.expired()) {
        expired.push_back(it->first);
      } else {
        remained.push(it->first);
      }
      queue_.pop();
    }
    CHECK(queue_.empty());
    CHECK_EQ(remained.size() + expired.size(), container_.size());

    for (auto const& key : expired) {
      container_.erase(key);
    }
    while (!remained.empty()) {
      auto p_fmat = remained.front();
      queue_.push(p_fmat);
      remained.pop();
    }
    this->CheckConsistent();
  }

  void ClearExcess() {
    this->CheckConsistent();
    // clear half of the entries to prevent repeatingly clearing cache.
    std::size_t half_size = max_size_ / 2;
    while (queue_.size() >= half_size && !queue_.empty()) {
      auto p_fmat = queue_.front();
      queue_.pop();
      container_.erase(p_fmat);
    }
    this->CheckConsistent();
  }

 public:
  /**
   * \param cache_size Maximum size of the cache.
   */
  explicit DMatrixCache(std::size_t cache_size) : max_size_{cache_size} {}

  DMatrixCache& operator=(DMatrixCache&& that) {
    CHECK(lock_.try_lock());
    lock_.unlock();
    CHECK(that.lock_.try_lock());
    that.lock_.unlock();
    std::swap(this->container_, that.container_);
    std::swap(this->queue_, that.queue_);
    std::swap(this->max_size_, that.max_size_);
    return *this;
  }

  /**
   * \brief Cache a new DMatrix if it's not in the cache already.
   *
   *  Passing in a `shared_ptr` is critical here.  First to create a `weak_ptr` inside the
   *  entry this shared pointer is necessary.  More importantly, the life time of this
   *  cache is tied to the shared pointer.
   *
   * \param m    shared pointer to the DMatrix that needs to be cached.
   * \param args The arguments for constructing a new cache item, if needed.
   *
   * \return The cache entry for passed in DMatrix, either an existing cache or newly
   *         created.
   */
  template <typename... Args>
  std::shared_ptr<CacheT> CacheItem(std::shared_ptr<DMatrix> m, Args const&... args) {
    CHECK(m);
    std::lock_guard<std::mutex> guard{lock_};

    this->ClearExpired();
    if (container_.size() >= max_size_) {
      this->ClearExcess();
    }
    // after clear, cache size < max_size
    CHECK_LT(container_.size(), max_size_);
    auto key = Key{m.get(), std::this_thread::get_id()};
    auto it = container_.find(key);
    if (it == container_.cend()) {
      // after the new DMatrix, cache size is at most max_size
      container_.emplace(key, Item{m, std::make_shared<CacheT>(args...)});
      queue_.emplace(key);
    }
    return container_.at(key).value;
  }
  /**
   * \brief Re-initialize the item in cache.
   *
   *   Since the shared_ptr is used to hold the item, any reference that lives outside of
   *   the cache can no-longer be reached from the cache.
   *
   *   We use reset instead of erase to avoid walking through the whole cache for renewing
   *   a single item. (the cache is FIFO, needs to maintain the order).
   */
  template <typename... Args>
  std::shared_ptr<CacheT> ResetItem(std::shared_ptr<DMatrix> m, Args const&... args) {
    std::lock_guard<std::mutex> guard{lock_};
    CheckConsistent();
    auto key = Key{m.get(), std::this_thread::get_id()};
    auto it = container_.find(key);
    CHECK(it != container_.cend());
    it->second = {m, std::make_shared<CacheT>(args...)};
    CheckConsistent();
    return it->second.value;
  }
  /**
   * \brief Get a const reference to the underlying hash map.  Clear expired caches before
   *        returning.
   */
  decltype(container_) const& Container() {
    std::lock_guard<std::mutex> guard{lock_};

    this->ClearExpired();
    return container_;
  }

  std::shared_ptr<CacheT> Entry(DMatrix const* m) const {
    std::lock_guard<std::mutex> guard{lock_};
    auto key = Key{m, std::this_thread::get_id()};
    CHECK(container_.find(key) != container_.cend());
    CHECK(!container_.at(key).ref.expired());
    return container_.at(key).value;
  }
};
}  // namespace xgboost
#endif  // XGBOOST_CACHE_H_
