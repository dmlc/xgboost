/**
 * Copyright 2023 by XGBoost contributors
 */
#ifndef XGBOOST_CACHE_H_
#define XGBOOST_CACHE_H_

#include <xgboost/logging.h>  // CHECK_EQ

#include <cstddef>            // std::size_t
#include <memory>             // std::weak_ptr,std::shared_ptr,std::make_shared
#include <queue>              // std:queue
#include <unordered_map>      // std::unordered_map
#include <vector>             // std::vector

namespace xgboost {
class DMatrix;
/**
 * \brief FIFO cache for DMatrix related data.
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
  };

  static constexpr std::size_t DefaultSize() { return 32; }

 protected:
  std::unordered_map<DMatrix const*, Item> container_;
  std::queue<DMatrix const*> queue_;
  std::size_t max_size_;

  void CheckConsistent() const { CHECK_EQ(queue_.size(), container_.size()); }

  void ClearExpired() {
    // Clear expired entries
    this->CheckConsistent();
    std::vector<DMatrix const*> expired;
    std::queue<DMatrix const*> remained;

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

    for (auto const* p_fmat : expired) {
      container_.erase(p_fmat);
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
    while (queue_.size() >= max_size_) {
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
  /**
   * \brief Cache a new DMatrix if it's no in the cache already.
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
  std::shared_ptr<CacheT>& CacheItem(std::shared_ptr<DMatrix> m, Args const&... args) {
    CHECK(m);
    this->ClearExpired();
    if (container_.size() >= max_size_) {
      this->ClearExcess();
    }
    // after clear, cache size < max_size
    CHECK_LT(container_.size(), max_size_);
    auto it = container_.find(m.get());
    if (it == container_.cend()) {
      // after the new DMatrix, cache size is at most max_size
      container_[m.get()] = {m, std::make_shared<CacheT>(args...)};
      queue_.push(m.get());
    }
    return container_.at(m.get()).value;
  }
  /**
   * \brief Get a const reference to the underlying hash map.  Clear expired caches before
   *        returning.
   */
  decltype(container_) const& Container() {
    this->ClearExpired();
    return container_;
  }

  std::shared_ptr<CacheT> Entry(DMatrix const* m) const {
    CHECK(container_.find(m) != container_.cend());
    CHECK(!container_.at(m).ref.expired());
    return container_.at(m).value;
  }
};
}  // namespace xgboost
#endif  // XGBOOST_CACHE_H_
