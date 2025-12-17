/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include "../common/linalg_op.h"
#include "array_page.h"
#include "sparse_page_source.h"

namespace xgboost::data {

struct ArrayPageNoOpWriter {};

struct ArrayPageReader {
  std::int32_t batch_idx = -1;
  std::vector<bst_idx_t> batch_ptr_;
  std::shared_ptr<ArrayPage> cache_;

  explicit ArrayPageReader(std::uint64_t offset_bytes, std::vector<bst_idx_t> const& batch_ptr,
                           std::shared_ptr<ArrayPage>);

  void Read(ArrayPage* page) const;
};

struct ArrayPageFormat {
  template <typename... Args>
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
  std::vector<bst_idx_t> batch_ptr_;

 public:
  using WriterT = ArrayPageNoOpWriter;
  using ReaderT = ArrayPageReader;
  using FormatT = ArrayPageFormat;

 public:
  std::unique_ptr<WriterT> CreateWriter(StringView, std::uint32_t) {
    return std::make_unique<WriterT>();
  }

  std::unique_ptr<ReaderT> CreateReader(StringView, std::uint64_t offset, std::uint64_t) const {
    // fixme: no copy
    CHECK(this->cache_);
    return std::make_unique<ReaderT>(offset, this->batch_ptr_, this->cache_);
  }

  auto CreatePageFormat(BatchParam const&) const {
    std::unique_ptr<FormatT> fmt{std::make_unique<ArrayPageFormat>()};
    return fmt;
  }

  void SetArrayCache(std::shared_ptr<ArrayPage> cache, std::vector<bst_idx_t> batch_ptr) {
    std::cout << "set cache:" << this << std::endl;
    this->cache_ = std::move(cache);
    this->batch_ptr_ = std::move(batch_ptr);
    CHECK(this->cache_);
  }
};

class ArrayPageSource : public SparsePageSourceImpl<ArrayPage, ArrayPageFormatPolicy> {
  using Super = SparsePageSourceImpl<ArrayPage, ArrayPageFormatPolicy>;
  std::vector<bst_idx_t> batch_ptr_;
  std::shared_ptr<ArrayPage> cache_;

 protected:
  void Fetch() final;
  void EndIter() final;
  ArrayPageSource& operator++() final;

 public:
  explicit ArrayPageSource(std::shared_ptr<ArrayPage> cache, std::vector<bst_idx_t> batch_ptr,
                           bst_feature_t n_features, std::shared_ptr<Cache> cache_info)
      : Super::SparsePageSourceImpl{std::numeric_limits<float>::quiet_NaN(), 2, n_features,
                                    std::move(cache_info)},
        batch_ptr_{std::move(batch_ptr)},
        cache_{cache} {
    std::cout << "create:" << this << std::endl;
    this->SetArrayCache(cache, this->batch_ptr_);
    this->Fetch();
  }
};

class ArrayCache {
  common::ThreadPool workers_{StringView{"aqu"}, 1, InitNewThread{}};
  std::shared_ptr<ArrayPage> cache_;
  std::size_t offset_{0};
  std::int32_t it_{0};
  std::future<void> last_;

  void Clear();

 public:
  explicit ArrayCache(Context const* ctx, common::Span<std::size_t const, 2> shape);

  void Push(std::shared_ptr<ArrayPage> page);
  std::shared_ptr<ArrayPage> Commit();
};
}  // namespace xgboost::data
