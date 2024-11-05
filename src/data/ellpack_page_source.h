/**
 * Copyright 2019-2024, XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_

#include <cstdint>  // for int32_t
#include <limits>   // for numeric_limits
#include <memory>   // for shared_ptr
#include <utility>  // for move
#include <vector>   // for vector

#include "../common/cuda_rt_utils.h"  // for SupportsPageableMem, SupportsAts
#include "../common/hist_util.h"      // for HistogramCuts
#include "ellpack_page.h"             // for EllpackPage
#include "ellpack_page_raw_format.h"  // for EllpackPageRawFormat
#include "sparse_page_source.h"       // for PageSourceIncMixIn
#include "xgboost/base.h"             // for bst_idx_t
#include "xgboost/context.h"          // for DeviceOrd
#include "xgboost/data.h"             // for BatchParam
#include "xgboost/span.h"             // for Span

namespace xgboost::data {
struct EllpackCacheInfo {
  BatchParam param;
  bool prefer_device{false};  // Prefer to cache the page in the device memory instead of host.
  std::int64_t max_num_device_pages{0};  // Maximum number of pages cached in device.
  float missing{std::numeric_limits<float>::quiet_NaN()};
  std::vector<bst_idx_t> cache_mapping;
  std::vector<bst_idx_t> buffer_bytes;
  std::vector<bst_idx_t> buffer_rows;

  EllpackCacheInfo() = default;
  EllpackCacheInfo(BatchParam param, bool prefer_device, std::int64_t max_num_device_pages,
                   float missing)
      : param{std::move(param)},
        prefer_device{prefer_device},
        max_num_device_pages{max_num_device_pages},
        missing{missing} {}
};

// We need to decouple the storage and the view of the storage so that we can implement
// concurrent read. As a result, there are two classes, one for cache storage, another one
// for stream.
//
// This is a memory-based cache. It can be a mixed of the device memory and the host memory.
struct EllpackMemCache {
  std::vector<std::unique_ptr<EllpackPageImpl>> pages;
  std::vector<std::size_t> offsets;
  // Size of each batch before concatenation.
  std::vector<bst_idx_t> sizes_orig;
  // Mapping of pages before concatenation to after concatenation.
  std::vector<std::size_t> const cache_mapping;
  // Cache info
  std::vector<std::size_t> const buffer_bytes;
  std::vector<bst_idx_t> const buffer_rows;
  bool const prefer_device;
  std::int64_t const max_num_device_pages;

  explicit EllpackMemCache(EllpackCacheInfo cinfo);
  ~EllpackMemCache();

  // The number of bytes for the entire cache.
  [[nodiscard]] std::size_t SizeBytes() const;

  [[nodiscard]] bool Empty() const { return this->SizeBytes() == 0; }

  [[nodiscard]] bst_idx_t NumBatchesOrig() const { return cache_mapping.size(); }
  [[nodiscard]] EllpackPageImpl const* At(std::int32_t k) const;

  [[nodiscard]] std::int64_t NumDevicePages() const;
};

// Pimpl to hide CUDA calls from the host compiler.
class EllpackHostCacheStreamImpl;

/**
 * @brief A view of the actual cache implemented by `EllpackHostCache`.
 */
class EllpackHostCacheStream {
  std::unique_ptr<EllpackHostCacheStreamImpl> p_impl_;

 public:
  explicit EllpackHostCacheStream(std::shared_ptr<EllpackMemCache> cache);
  ~EllpackHostCacheStream();
  /**
   * @brief Get a shared handler to the cache.
   */
  std::shared_ptr<EllpackMemCache const> Share() const;
  /**
   * @brief Stream seek.
   *
   * @param offset_bytes This must align to the actual cached page size.
   */
  void Seek(bst_idx_t offset_bytes);
  /**
   * @brief Read a page from the cache.
   *
   * The read page might be concatenated during page write.
   *
   * @param page[out] The returned page.
   * @param prefetch_copy[in] Does the stream need to copy the page?
   */
  void Read(EllpackPage* page, bool prefetch_copy) const;
  /**
   * @brief Add a new page to the host cache.
   *
   * This method might append the input page to a previously stored page to increase
   * individual page size.
   *
   * @return Whether a new cache page is create. False if the new page is appended to the
   * previous one.
   */
  [[nodiscard]] bool Write(EllpackPage const& page);
};

template <typename S>
class EllpackFormatPolicy {
  std::shared_ptr<common::HistogramCuts const> cuts_{nullptr};
  DeviceOrd device_;
  bool has_hmm_{curt::SupportsPageableMem()};

  EllpackCacheInfo cache_info_;
  static_assert(std::is_same_v<S, EllpackPage>);

 public:
  using FormatT = EllpackPageRawFormat;

 public:
  EllpackFormatPolicy() {
    StringView msg{" The overhead of iterating through external memory might be significant."};
    if (!has_hmm_) {
      LOG(WARNING) << "CUDA heterogeneous memory management is not available." << msg;
    } else if (!curt::SupportsAts()) {
      LOG(WARNING) << "CUDA address translation service is not available." << msg;
    }
#if !defined(XGBOOST_USE_RMM)
    LOG(WARNING) << "XGBoost is not built with RMM support." << msg;
#endif
    if (!GlobalConfigThreadLocalStore::Get()->use_rmm) {
      LOG(WARNING) << "`use_rmm` is set to false." << msg;
    }
    std::int32_t major{0}, minor{0};
    curt::DrVersion(&major, &minor);
    if ((major < 12 || (major == 12 && minor < 7)) && curt::SupportsAts()) {
      // Use ATS, but with an old kernel driver.
      LOG(WARNING) << "Using an old kernel driver with supported CTK<12.7."
                   << "The latest version of CTK supported by the current driver: " << major << "."
                   << minor << "." << msg;
    }
  }
  // For testing with the HMM flag.
  explicit EllpackFormatPolicy(bool has_hmm) : has_hmm_{has_hmm} {}

  [[nodiscard]] auto CreatePageFormat(BatchParam const& param) const {
    CHECK_EQ(cuts_->cut_values_.Device(), device_);
    std::unique_ptr<FormatT> fmt{new EllpackPageRawFormat{cuts_, device_, param, has_hmm_}};
    return fmt;
  }
  void SetCuts(std::shared_ptr<common::HistogramCuts const> cuts, DeviceOrd device,
               EllpackCacheInfo cinfo) {
    std::swap(this->cuts_, cuts);
    this->device_ = device;
    CHECK(this->device_.IsCUDA());
    this->cache_info_ = std::move(cinfo);
  }
  [[nodiscard]] auto GetCuts() const {
    CHECK(cuts_);
    return cuts_;
  }
  [[nodiscard]] auto Device() const { return this->device_; }
  [[nodiscard]] auto const& CacheInfo() { return this->cache_info_; }
};

template <typename S, template <typename> typename F>
class EllpackCacheStreamPolicy : public F<S> {
  std::shared_ptr<EllpackMemCache> p_cache_;

 public:
  using WriterT = EllpackHostCacheStream;
  using ReaderT = EllpackHostCacheStream;

 public:
  EllpackCacheStreamPolicy() = default;
  [[nodiscard]] std::unique_ptr<WriterT> CreateWriter(StringView name, std::uint32_t iter);

  [[nodiscard]] std::unique_ptr<ReaderT> CreateReader(StringView name, bst_idx_t offset,
                                                      bst_idx_t length) const;
};

template <typename S, template <typename> typename F>
class EllpackMmapStreamPolicy : public F<S> {
  bool has_hmm_{curt::SupportsPageableMem()};

 public:
  using WriterT = common::AlignedFileWriteStream;
  using ReaderT = common::AlignedResourceReadStream;

 public:
  EllpackMmapStreamPolicy() = default;
  // For testing with the HMM flag.
  template <
      typename std::enable_if_t<std::is_same_v<F<S>, EllpackFormatPolicy<EllpackPage>>>* = nullptr>
  explicit EllpackMmapStreamPolicy(bool has_hmm) : F<S>{has_hmm}, has_hmm_{has_hmm} {}

  [[nodiscard]] std::unique_ptr<WriterT> CreateWriter(StringView name, std::uint32_t iter) {
    std::unique_ptr<common::AlignedFileWriteStream> fo;
    if (iter == 0) {
      fo = std::make_unique<common::AlignedFileWriteStream>(name, "wb");
    } else {
      fo = std::make_unique<common::AlignedFileWriteStream>(name, "ab");
    }
    return fo;
  }

  [[nodiscard]] std::unique_ptr<ReaderT> CreateReader(StringView name, bst_idx_t offset,
                                                      bst_idx_t length) const;
};

/**
 * @brief Calculate the size of each internal cached page along with the mapping of old
 *        pages to the new pages.
 */
void CalcCacheMapping(Context const* ctx, bool is_dense,
                      std::shared_ptr<common::HistogramCuts const> cuts,
                      std::int64_t min_cache_page_bytes, ExternalDataInfo const& ext_info,
                      EllpackCacheInfo* cinfo);

/**
 * @brief Ellpack source with sparse pages as the underlying source.
 */
template <typename F>
class EllpackPageSourceImpl : public PageSourceIncMixIn<EllpackPage, F> {
  using Super = PageSourceIncMixIn<EllpackPage, F>;
  bool is_dense_;
  bst_idx_t row_stride_;
  BatchParam param_;
  common::Span<FeatureType const> feature_types_;

 public:
  EllpackPageSourceImpl(Context const* ctx, bst_feature_t n_features, std::size_t n_batches,
                        std::shared_ptr<Cache> cache, std::shared_ptr<common::HistogramCuts> cuts,
                        bool is_dense, bst_idx_t row_stride,
                        common::Span<FeatureType const> feature_types,
                        std::shared_ptr<SparsePageSource> source, EllpackCacheInfo const& cinfo)
      : Super{cinfo.missing, ctx->Threads(), n_features, n_batches, cache, false},
        is_dense_{is_dense},
        row_stride_{row_stride},
        param_{std::move(cinfo.param)},
        feature_types_{feature_types} {
    this->source_ = source;
    cuts->SetDevice(ctx->Device());
    this->SetCuts(std::move(cuts), ctx->Device(), cinfo);
    this->Fetch();
  }

  void Fetch() final;
};

// Cache to host
using EllpackPageHostSource =
    EllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

// Cache to disk
using EllpackPageSource =
    EllpackPageSourceImpl<EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

/**
 * @brief Ellpack source directly interfaces with user-defined iterators.
 */
template <typename FormatCreatePolicy>
class ExtEllpackPageSourceImpl : public ExtQantileSourceMixin<EllpackPage, FormatCreatePolicy> {
  using Super = ExtQantileSourceMixin<EllpackPage, FormatCreatePolicy>;

  Context const* ctx_;
  BatchParam p_;
  DMatrixProxy* proxy_;
  MetaInfo* info_;
  ExternalDataInfo ext_info_;

 public:
  ExtEllpackPageSourceImpl(
      Context const* ctx, MetaInfo* info, ExternalDataInfo ext_info, std::shared_ptr<Cache> cache,
      std::shared_ptr<common::HistogramCuts> cuts,
      std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> source,
      DMatrixProxy* proxy, EllpackCacheInfo const& cinfo)
      : Super{cinfo.missing, ctx->Threads(), static_cast<bst_feature_t>(info->num_col_), source,
              cache},
        ctx_{ctx},
        p_{cinfo.param},
        proxy_{proxy},
        info_{info},
        ext_info_{std::move(ext_info)} {
    cuts->SetDevice(ctx->Device());
    this->SetCuts(std::move(cuts), ctx->Device(), cinfo);
    CHECK(!this->cache_info_->written);
    this->source_->Reset();
    CHECK(this->source_->Next());
    this->Fetch();
  }

  void Fetch() final;
  // Need a specialized end iter as we can concatenate pages.
  void EndIter() final {
    if (this->cache_info_->written) {
      CHECK_EQ(this->Iter(), this->cache_info_->Size());
    } else {
      CHECK_LE(this->cache_info_->Size(), this->ext_info_.n_batches);
    }
    this->cache_info_->Commit();
    CHECK_GE(this->count_, 1);
    this->count_ = 0;
  }
};

// Cache to host
using ExtEllpackPageHostSource =
    ExtEllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

// Cache to disk
using ExtEllpackPageSource =
    ExtEllpackPageSourceImpl<EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

#if !defined(XGBOOST_USE_CUDA)
template <typename F>
inline void EllpackPageSourceImpl<F>::Fetch() {
  // silent the warning about unused variables.
  (void)(row_stride_);
  (void)(is_dense_);
  common::AssertGPUSupport();
}

template <typename F>
inline void ExtEllpackPageSourceImpl<F>::Fetch() {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data

#endif  // XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
