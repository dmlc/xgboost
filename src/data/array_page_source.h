/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include "array_page.h"
#include "sparse_page_source.h"

namespace xgboost::data {
class ArrayPageSource : public SparsePageSourceImpl<ArrayPage> {
  using Super = SparsePageSourceImpl<ArrayPage>;

 protected:
  void Fetch() final;
  void EndIter() final;
  ArrayPageSource& operator++() final;

 public:
  explicit ArrayPageSource(std::shared_ptr<Cache> cache)
      : Super::SparsePageSourceImpl{std::numeric_limits<float>::quiet_NaN(), 2, 1u,
                                    std::move(cache)} {}
};

class ArrayCache {
  common::ThreadPool workers_{StringView{"aqu"}, 1, InitNewThread{}};
  ArrayPage cache_;
  std::size_t offset_{0};
  std::int32_t it_{0};
  std::future<void> last_;

  void Clear();

 public:
  explicit ArrayCache(Context const* ctx, common::Span<std::size_t const, 2> shape);

  void Push(std::shared_ptr<ArrayPage> page);
  ArrayPage Commit();
};
}  // namespace xgboost::data
