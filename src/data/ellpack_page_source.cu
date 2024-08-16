/**
 * Copyright 2019-2024, XGBoost contributors
 */
#include <thrust/host_vector.h>  // for host_vector

#include <cstddef>  // for size_t
#include <cstdint>  // for int8_t, uint64_t, uint32_t
#include <memory>   // for shared_ptr, make_unique, make_shared
#include <utility>  // for move

#include "../common/common.h"                 // for safe_cuda
#include "../common/cuda_pinned_allocator.h"  // for pinned_allocator
#include "../common/device_helpers.cuh"       // for CUDAStreamView, DefaultStream
#include "../common/resource.cuh"             // for PrivateCudaMmapConstStream
#include "ellpack_page.cuh"                   // for EllpackPageImpl
#include "ellpack_page.h"                     // for EllpackPage
#include "ellpack_page_source.h"
#include "proxy_dmatrix.cuh"  // for Dispatch
#include "xgboost/base.h"     // for bst_idx_t

namespace xgboost::data {
struct EllpackHostCache {
  thrust::host_vector<std::int8_t, common::cuda::pinned_allocator<std::int8_t>> cache;

  void Resize(std::size_t n, dh::CUDAStreamView stream) {
    stream.Sync();  // Prevent partial copy inside resize.
    cache.resize(n);
  }
};

class EllpackHostCacheStreamImpl {
  std::shared_ptr<EllpackHostCache> cache_;
  bst_idx_t cur_ptr_{0};
  bst_idx_t bound_{0};

 public:
  explicit EllpackHostCacheStreamImpl(std::shared_ptr<EllpackHostCache> cache)
      : cache_{std::move(cache)} {}

  [[nodiscard]] bst_idx_t Write(void const* ptr, bst_idx_t n_bytes) {
    auto n = cur_ptr_ + n_bytes;
    if (n > cache_->cache.size()) {
      cache_->Resize(n, dh::DefaultStream());
    }
    dh::safe_cuda(cudaMemcpyAsync(cache_->cache.data() + cur_ptr_, ptr, n_bytes, cudaMemcpyDefault,
                                  dh::DefaultStream()));
    cur_ptr_ = n;
    return n_bytes;
  }

  [[nodiscard]] bool Read(void* ptr, bst_idx_t n_bytes) {
    CHECK_LE(cur_ptr_ + n_bytes, bound_);
    dh::safe_cuda(cudaMemcpyAsync(ptr, cache_->cache.data() + cur_ptr_, n_bytes, cudaMemcpyDefault,
                                  dh::DefaultStream()));
    cur_ptr_ += n_bytes;
    return true;
  }

  [[nodiscard]] bst_idx_t Tell() const { return cur_ptr_; }
  void Seek(bst_idx_t offset_bytes) { cur_ptr_ = offset_bytes; }
  void Bound(bst_idx_t offset_bytes) {
    CHECK_LE(offset_bytes, cache_->cache.size());
    this->bound_ = offset_bytes;
  }
};

/**
 * EllpackHostCacheStream
 */

EllpackHostCacheStream::EllpackHostCacheStream(std::shared_ptr<EllpackHostCache> cache)
    : p_impl_{std::make_unique<EllpackHostCacheStreamImpl>(std::move(cache))} {}

EllpackHostCacheStream::~EllpackHostCacheStream() = default;

[[nodiscard]] bst_idx_t EllpackHostCacheStream::Write(void const* ptr, bst_idx_t n_bytes) {
  return this->p_impl_->Write(ptr, n_bytes);
}

[[nodiscard]] bool EllpackHostCacheStream::Read(void* ptr, bst_idx_t n_bytes) {
  return this->p_impl_->Read(ptr, n_bytes);
}

[[nodiscard]] bst_idx_t EllpackHostCacheStream::Tell() const { return this->p_impl_->Tell(); }

void EllpackHostCacheStream::Seek(bst_idx_t offset_bytes) { this->p_impl_->Seek(offset_bytes); }

void EllpackHostCacheStream::Bound(bst_idx_t offset_bytes) { this->p_impl_->Bound(offset_bytes); }

/**
 * EllpackCacheStreamPolicy
 */

template <typename S, template <typename> typename F>
EllpackCacheStreamPolicy<S, F>::EllpackCacheStreamPolicy()
    : p_cache_{std::make_shared<EllpackHostCache>()} {}

template <typename S, template <typename> typename F>
[[nodiscard]] std::unique_ptr<typename EllpackCacheStreamPolicy<S, F>::WriterT>
EllpackCacheStreamPolicy<S, F>::CreateWriter(StringView, std::uint32_t iter) {
  auto fo = std::make_unique<EllpackHostCacheStream>(this->p_cache_);
  if (iter == 0) {
    CHECK(this->p_cache_->cache.empty());
  } else {
    fo->Seek(this->p_cache_->cache.size());
  }
  return fo;
}

template <typename S, template <typename> typename F>
[[nodiscard]] std::unique_ptr<typename EllpackCacheStreamPolicy<S, F>::ReaderT>
EllpackCacheStreamPolicy<S, F>::CreateReader(StringView, bst_idx_t offset, bst_idx_t length) const {
  auto fi = std::make_unique<ReaderT>(this->p_cache_);
  fi->Seek(offset);
  fi->Bound(offset + length);
  CHECK_EQ(fi->Tell(), offset);
  return fi;
}

// Instantiation
template EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::EllpackCacheStreamPolicy();

template std::unique_ptr<
    typename EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::WriterT>
EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::CreateWriter(StringView name,
                                                                         std::uint32_t iter);

template std::unique_ptr<
    typename EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::ReaderT>
EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::CreateReader(
    StringView name, std::uint64_t offset, std::uint64_t length) const;

/**
 * EllpackMmapStreamPolicy
 */

template <typename S, template <typename> typename F>
[[nodiscard]] std::unique_ptr<typename EllpackMmapStreamPolicy<S, F>::ReaderT>
EllpackMmapStreamPolicy<S, F>::CreateReader(StringView name, bst_idx_t offset,
                                            bst_idx_t length) const {
  if (has_hmm_) {
    return std::make_unique<common::PrivateCudaMmapConstStream>(name, offset, length);
  } else {
    return std::make_unique<common::PrivateMmapConstStream>(name, offset, length);
  }
}

// Instantiation
template std::unique_ptr<
    typename EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>::ReaderT>
EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>::CreateReader(StringView name,
                                                                        bst_idx_t offset,
                                                                        bst_idx_t length) const;

/**
 * EllpackPageSourceImpl
 */
template <typename F>
void EllpackPageSourceImpl<F>::Fetch() {
  dh::safe_cuda(cudaSetDevice(this->Device().ordinal));
  if (!this->ReadCache()) {
    if (this->count_ != 0 && !this->sync_) {
      // source is initialized to be the 0th page during construction, so when count_ is 0
      // there's no need to increment the source.
      ++(*this->source_);
    }
    // This is not read from cache so we still need it to be synced with sparse page source.
    CHECK_EQ(this->count_, this->source_->Iter());
    auto const& csr = this->source_->Page();
    this->page_.reset(new EllpackPage{});
    auto* impl = this->page_->Impl();
    Context ctx = Context{}.MakeCUDA(this->Device().ordinal);
    *impl = EllpackPageImpl{&ctx, this->GetCuts(), *csr, is_dense_, row_stride_, feature_types_};
    this->page_->SetBaseRowId(csr->base_rowid);
    LOG(INFO) << "Generated an Ellpack page with size: " << impl->MemCostBytes()
              << " from a SparsePage with size:" << csr->MemCostBytes();
    this->WriteCache();
  }
}

// Instantiation
template void
EllpackPageSourceImpl<DefaultFormatStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
EllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
EllpackPageSourceImpl<EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();

/**
 * ExtEllpackPageSourceImpl
 */
template <typename F>
void ExtEllpackPageSourceImpl<F>::Fetch() {
  dh::safe_cuda(cudaSetDevice(this->Device().ordinal));
  if (!this->ReadCache()) {
    auto iter = this->source_->Iter();
    CHECK_EQ(this->count_, iter);
    ++(*this->source_);
    CHECK_GE(this->source_->Iter(), 1);
    cuda_impl::Dispatch(proxy_, [this](auto const& value) {
      proxy_->Info().feature_types.SetDevice(dh::GetDevice(this->ctx_));
      auto d_feature_types = proxy_->Info().feature_types.ConstDeviceSpan();
      auto n_samples = value.NumRows();

      dh::device_vector<size_t> row_counts(n_samples + 1, 0);
      common::Span<size_t> row_counts_span(row_counts.data().get(), row_counts.size());
      cuda_impl::Dispatch(proxy_, [=](auto const& value) {
        return GetRowCounts(value, row_counts_span, dh::GetDevice(this->ctx_), this->missing_);
      });

      this->page_.reset(new EllpackPage{});
      *this->page_->Impl() = EllpackPageImpl{this->ctx_,
                                             value,
                                             this->missing_,
                                             this->info_->IsDense(),
                                             row_counts_span,
                                             d_feature_types,
                                             this->ext_info_.row_stride,
                                             n_samples,
                                             this->GetCuts()};
      this->info_->Extend(proxy_->Info(), false, true);
    });
    this->page_->SetBaseRowId(this->ext_info_.base_rows.at(iter));
    this->WriteCache();
  }
}

// Instantiation
template void
ExtEllpackPageSourceImpl<DefaultFormatStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
ExtEllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
ExtEllpackPageSourceImpl<EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
}  // namespace xgboost::data
