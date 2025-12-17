/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include "array_page.h"
#include "sparse_page_source.h"

namespace xgboost::data {

struct ArrayPageNoOpWriter {};

struct ArrayPageReader {};

struct ArrayPageFormat {
  template <typename... Args>
  std::size_t Write(ArrayPage const& page, ArrayPageNoOpWriter*) const {
    return page.gpairs.Data()->SizeBytes();
  }
  bool Read(ArrayPage* page, ArrayPageReader* fi) const {
    return true;
  }
};

struct ArrayPageFormatPolicy {
 private:
  ArrayPage cache_;

 public:
  using WriterT = ArrayPageNoOpWriter;
  using ReaderT = ArrayPageReader;
  using FormatT = ArrayPageFormat;

 public:
  std::unique_ptr<WriterT> CreateWriter(StringView, std::uint32_t) {
    return std::make_unique<WriterT>();
  }

  std::unique_ptr<ReaderT> CreateReader(StringView name, std::uint64_t offset,
                                        std::uint64_t length) const {
    return std::make_unique<ReaderT>();
  }

  auto CreatePageFormat(BatchParam const&) const {
    std::unique_ptr<FormatT> fmt{std::make_unique<ArrayPageFormat>()};
    return fmt;
  }

  void SetCache(ArrayPage cache) { this->cache_ = std::move(cache); }
};

class ArrayPageSource : public SparsePageSourceImpl<ArrayPage, ArrayPageFormatPolicy> {
  using Super = SparsePageSourceImpl<ArrayPage, ArrayPageFormatPolicy>;
  std::vector<bst_idx_t> batch_ptr_;

 protected:
  void Fetch() final;
  void EndIter() final;
  ArrayPageSource& operator++() final;

 public:
  explicit ArrayPageSource(ArrayPage cache, std::vector<bst_idx_t> batch_ptr,
                           std::shared_ptr<Cache> cache_info)
      : Super::SparsePageSourceImpl{std::numeric_limits<float>::quiet_NaN(), 2, 1u,
                                    std::move(cache_info)},
        batch_ptr_{std::move(batch_ptr)} {
    this->SetCache(std::move(cache));
  }
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
