/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include "array_page.h"
#include "sparse_page_source.h"
#include "../common/linalg_op.h"

namespace xgboost::data {

struct ArrayPageNoOpWriter {};

struct ArrayPageReader {
  std::int32_t batch_idx = 0;
  std::vector<bst_idx_t> batch_ptr_;
  std::shared_ptr<ArrayPage> cache;

  explicit ArrayPageReader(std::uint64_t offset_bytes, std::vector<bst_idx_t> const& batch_ptr) {
    CHECK(cache);
    // fixme: binary seaerch
    auto size_bytes = [&](std::size_t i) {
      // auto beg = batch_ptr.at(i);
      auto end = batch_ptr.at(i + 1);
      auto n_columns = this->cache->gpairs.Shape(1);
      //  - beg
      auto n_rows = end;
      auto n_bytes = common::SizeBytes<GradientPair>(n_rows * n_columns);
      // accumulated bytes
      return n_bytes;
    };
    for (std::size_t i = 0; i < batch_ptr.size(); ++i) {
      auto n_bytes = size_bytes(i);
      if (n_bytes == offset_bytes) {
        this->batch_idx = i;
        break;
      }
    }
    LOG(FATAL) << "Seek failed";
    this->batch_ptr_ = batch_ptr;
  }

  void Read(ArrayPage* page) const {
    auto begin = this->batch_ptr_.at(batch_idx);
    auto end = this->batch_ptr_.at(batch_idx + 1);
    auto h_cache =
        std::as_const(this->cache->gpairs).Slice(linalg::Range(begin, end), linalg::All());
    Context ctx = Context{}.MakeCUDA(0);  // fixme
    linalg::Copy(&ctx, h_cache, &page->gpairs);
  }
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
  ArrayPage cache_;
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
    return std::make_unique<ReaderT>(offset, this->batch_ptr_);
  }

  auto CreatePageFormat(BatchParam const&) const {
    std::unique_ptr<FormatT> fmt{std::make_unique<ArrayPageFormat>()};
    return fmt;
  }

  void SetCache(ArrayPage cache, std::vector<bst_idx_t> batch_ptr) {
    this->cache_ = std::move(cache);
    this->batch_ptr_ = std::move(batch_ptr);
  }
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
    this->SetCache(std::move(cache), this->batch_ptr_);
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
