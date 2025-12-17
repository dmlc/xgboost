/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include "array_page.h"
#include "sparse_page_source.h"

namespace xgboost::data {

struct ArrayPageNoOpWriter {};

class ArrayPageReader {
  std::int32_t batch_idx_ = -1;
  std::shared_ptr<ArrayPage> cache_;

 public:
  explicit ArrayPageReader(std::uint64_t offset_bytes, std::shared_ptr<ArrayPage> cache);

  void Read(ArrayPage* page) const;
};

struct ArrayPageFormat {
  std::size_t Write(ArrayPage const& page, ArrayPageNoOpWriter*) const {
    return page.gpairs.Data()->SizeBytes();
  }
  bool Read(ArrayPage* page, ArrayPageReader* fi) const {
    fi->Read(page);
    return true;
  }
};

struct ArrayPageFormatPolicy {
 private:
  std::shared_ptr<ArrayPage> cache_;

 public:
  using WriterT = ArrayPageNoOpWriter;
  using ReaderT = ArrayPageReader;
  using FormatT = ArrayPageFormat;

 public:
  auto CreateWriter(StringView, std::uint32_t) { return std::make_unique<WriterT>(); }

  [[nodiscard]] auto CreateReader(StringView, std::uint64_t offset, std::uint64_t) const {
    CHECK(this->cache_);
    return std::make_unique<ReaderT>(offset, this->cache_);
  }

  [[nodiscard]] auto CreatePageFormat(BatchParam const&) const {
    return std::make_unique<ArrayPageFormat>();
  }

  void SetArrayCache(std::shared_ptr<ArrayPage> cache) {
    this->cache_ = std::move(cache);
    CHECK(this->cache_);
  }
};

class ArrayPageSource : public SparsePageSourceImpl<ArrayPage, ArrayPageFormatPolicy> {
  using Super = SparsePageSourceImpl<ArrayPage, ArrayPageFormatPolicy>;
  std::shared_ptr<ArrayPage> cache_;

 protected:
  void Fetch() final;
  void EndIter() final;
  ArrayPageSource& operator++() final;

 public:
  explicit ArrayPageSource(std::shared_ptr<ArrayPage> cache, bst_feature_t n_features,
                           std::shared_ptr<Cache> cache_info)
      : Super::SparsePageSourceImpl{std::numeric_limits<float>::quiet_NaN(), 2, n_features,
                                    std::move(cache_info)},
        cache_{cache} {
    this->SetArrayCache(cache);
    this->Fetch();
  }
};

class ArrayCacheWriter {
  // Single thread, we use the thread pool for easy task submit
  common::ThreadPool workers_{StringView{"aqu"}, 1, InitNewThread{}};
  std::shared_ptr<ArrayPage> cache_;
  std::size_t offset_{0};

 public:
  explicit ArrayCacheWriter(Context const* ctx, common::Span<std::size_t const, 2> shape);

  void Push(std::shared_ptr<ArrayPage> page);
  std::shared_ptr<ArrayPage> Commit();
};
}  // namespace xgboost::data
