/**
 * Copyright 2019-2024, XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_

#include <cstdint>  // for int32_t
#include <memory>   // for shared_ptr
#include <utility>  // for move
#include <vector>   // for vector

#include "../common/cuda_rt_utils.h"  // for SupportsPageableMem
#include "../common/hist_util.h"      // for HistogramCuts
#include "ellpack_page.h"             // for EllpackPage
#include "ellpack_page_raw_format.h"  // for EllpackPageRawFormat
#include "sparse_page_source.h"       // for PageSourceIncMixIn
#include "xgboost/base.h"             // for bst_idx_t
#include "xgboost/context.h"          // for DeviceOrd
#include "xgboost/data.h"             // for BatchParam
#include "xgboost/span.h"             // for Span

namespace xgboost::data {
// We need to decouple the storage and the view of the storage so that we can implement
// concurrent read.

// Dummy type to hide CUDA calls from the host compiler.
struct EllpackHostCache;
// Pimpl to hide CUDA calls from the host compiler.
class EllpackHostCacheStreamImpl;

// A view onto the actual cache implemented by `EllpackHostCache`.
class EllpackHostCacheStream {
  std::unique_ptr<EllpackHostCacheStreamImpl> p_impl_;

 public:
  explicit EllpackHostCacheStream(std::shared_ptr<EllpackHostCache> cache);
  ~EllpackHostCacheStream();

  [[nodiscard]] bst_idx_t Write(void const* ptr, bst_idx_t n_bytes);
  template <typename T>
  [[nodiscard]] std::enable_if_t<std::is_pod_v<T>, bst_idx_t> Write(T const& v) {
    return this->Write(&v, sizeof(T));
  }

  [[nodiscard]] bool Read(void* ptr, bst_idx_t n_bytes);

  template <typename T>
  [[nodiscard]] auto Read(T* ptr) -> std::enable_if_t<std::is_pod_v<T>, bool> {
    return this->Read(ptr, sizeof(T));
  }

  [[nodiscard]] bst_idx_t Tell() const;
  void Seek(bst_idx_t offset_bytes);
  // Limit the size of read. offset_bytes is the maximum offset that this stream can read
  // to. An error is raised if the limited is exceeded.
  void Bound(bst_idx_t offset_bytes);
};

template <typename S>
class EllpackFormatPolicy {
  std::shared_ptr<common::HistogramCuts const> cuts_{nullptr};
  DeviceOrd device_;
  bool has_hmm_{common::SupportsPageableMem()};

 public:
  using FormatT = EllpackPageRawFormat;

 public:
  EllpackFormatPolicy() = default;
  // For testing with the HMM flag.
  explicit EllpackFormatPolicy(bool has_hmm) : has_hmm_{has_hmm} {}

  [[nodiscard]] auto CreatePageFormat() const {
    CHECK_EQ(cuts_->cut_values_.Device(), device_);
    std::unique_ptr<FormatT> fmt{new EllpackPageRawFormat{cuts_, device_, has_hmm_}};
    return fmt;
  }

  void SetCuts(std::shared_ptr<common::HistogramCuts const> cuts, DeviceOrd device) {
    std::swap(cuts_, cuts);
    device_ = device;
    CHECK(this->device_.IsCUDA());
  }
  [[nodiscard]] auto GetCuts() {
    CHECK(cuts_);
    return cuts_;
  }
  [[nodiscard]] auto Device() const { return device_; }
};

template <typename S, template <typename> typename F>
class EllpackCacheStreamPolicy : public F<S> {
  std::shared_ptr<EllpackHostCache> p_cache_;

 public:
  using WriterT = EllpackHostCacheStream;
  using ReaderT = EllpackHostCacheStream;

 public:
  EllpackCacheStreamPolicy();
  [[nodiscard]] std::unique_ptr<WriterT> CreateWriter(StringView name, std::uint32_t iter);

  [[nodiscard]] std::unique_ptr<ReaderT> CreateReader(StringView name, bst_idx_t offset,
                                                      bst_idx_t length) const;
};

template <typename S, template <typename> typename F>
class EllpackMmapStreamPolicy : public F<S> {
  bool has_hmm_{common::SupportsPageableMem()};

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

template <typename F>
class EllpackPageSourceImpl : public PageSourceIncMixIn<EllpackPage, F> {
  using Super = PageSourceIncMixIn<EllpackPage, F>;
  bool is_dense_;
  bst_idx_t row_stride_;
  BatchParam param_;
  common::Span<FeatureType const> feature_types_;

 public:
  EllpackPageSourceImpl(float missing, std::int32_t nthreads, bst_feature_t n_features,
                        std::size_t n_batches, std::shared_ptr<Cache> cache, BatchParam param,
                        std::shared_ptr<common::HistogramCuts> cuts, bool is_dense,
                        bst_idx_t row_stride, common::Span<FeatureType const> feature_types,
                        std::shared_ptr<SparsePageSource> source, DeviceOrd device)
      : Super{missing, nthreads, n_features, n_batches, cache, false},
        is_dense_{is_dense},
        row_stride_{row_stride},
        param_{std::move(param)},
        feature_types_{feature_types} {
    this->source_ = source;
    cuts->SetDevice(device);
    this->SetCuts(std::move(cuts), device);
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

template <typename FormatCreatePolicy>
class ExtEllpackPageSourceImpl : public ExtQantileSourceMixin<EllpackPage, FormatCreatePolicy> {
  using Super = ExtQantileSourceMixin<EllpackPage, FormatCreatePolicy>;

  Context const* ctx_;
  BatchParam p_;
  DMatrixProxy* proxy_;
  MetaInfo* info_;
  ExternalDataInfo ext_info_;

  std::vector<bst_idx_t> base_rows_;

 public:
  ExtEllpackPageSourceImpl(
      Context const* ctx, float missing, MetaInfo* info, ExternalDataInfo ext_info,
      std::shared_ptr<Cache> cache, BatchParam param, std::shared_ptr<common::HistogramCuts> cuts,
      std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> source,
      DMatrixProxy* proxy, std::vector<bst_idx_t> base_rows)
      : Super{missing,
              ctx->Threads(),
              static_cast<bst_feature_t>(info->num_col_),
              ext_info.n_batches,
              source,
              cache},
        ctx_{ctx},
        p_{std::move(param)},
        proxy_{proxy},
        info_{info},
        ext_info_{std::move(ext_info)},
        base_rows_{std::move(base_rows)} {
    this->SetCuts(std::move(cuts), ctx->Device());
    this->Fetch();
  }

  void Fetch() final;
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
