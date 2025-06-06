/**
 * Copyright 2019-2025, XGBoost contributors
 */
#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for int8_t, uint64_t, uint32_t
#include <memory>     // for shared_ptr, make_unique, make_shared
#include <numeric>    // for accumulate
#include <utility>    // for move

#include "../common/common.h"               // for HumanMemUnit, safe_cuda
#include "../common/cuda_rt_utils.h"        // for SetDevice
#include "../common/cuda_stream_pool.cuh"   // for StreamPool
#include "../common/device_helpers.cuh"     // for CUDAStreamView, DefaultStream
#include "../common/ref_resource_view.cuh"  // for MakeFixedVecWithCudaMalloc
#include "../common/resource.cuh"           // for PrivateCudaMmapConstStream
#include "../common/transform_iterator.h"   // for MakeIndexTransformIter
#include "batch_utils.h"                    // for HostRatioIsAuto
#include "ellpack_page.cuh"                 // for EllpackPageImpl
#include "ellpack_page.h"                   // for EllpackPage
#include "ellpack_page_source.h"
#include "proxy_dmatrix.cuh"  // for Dispatch
#include "xgboost/base.h"     // for bst_idx_t

namespace xgboost::data {
/**
 * Cache
 */
EllpackMemCache::EllpackMemCache(EllpackCacheInfo cinfo, std::int32_t n_workers)
    : cache_mapping{std::move(cinfo.cache_mapping)},
      buffer_bytes{std::move(cinfo.buffer_bytes)},
      buffer_rows{std::move(cinfo.buffer_rows)},
      cache_host_ratio{cinfo.cache_host_ratio},
      streams{std::make_unique<curt::StreamPool>(n_workers)} {
  CHECK_EQ(buffer_bytes.size(), buffer_rows.size());
  CHECK(!detail::HostRatioIsAuto(this->cache_host_ratio));
  CHECK_GE(this->cache_host_ratio, 0.0) << error::CacheHostRatioInvalid();
  CHECK_LE(this->cache_host_ratio, 1.0) << error::CacheHostRatioInvalid();
}

EllpackMemCache::~EllpackMemCache() = default;

[[nodiscard]] std::size_t EllpackMemCache::SizeBytes() const noexcept(true) {
  auto it = common::MakeIndexTransformIter([&](auto i) { return this->SizeBytes(i); });
  using T = std::iterator_traits<decltype(it)>::value_type;
  return std::accumulate(it, it + this->Size(), static_cast<T>(0));
}

[[nodiscard]] std::size_t EllpackMemCache::DeviceSizeBytes() const noexcept(true) {
  auto it =
      common::MakeIndexTransformIter([&](auto i) { return this->d_pages.at(i).size_bytes(); });
  using T = std::iterator_traits<decltype(it)>::value_type;
  return std::accumulate(it, it + this->Size(), static_cast<T>(0));
}

[[nodiscard]] std::size_t EllpackMemCache::SizeBytes(std::size_t i) const noexcept(true) {
  return this->h_pages.at(i)->MemCostBytes() + this->d_pages.at(i).size_bytes();
}

[[nodiscard]] std::size_t EllpackMemCache::GidxSizeBytes(std::size_t i) const noexcept(true) {
  return this->h_pages.at(i)->gidx_buffer.size_bytes() + this->d_pages.at(i).size_bytes();
}

[[nodiscard]] std::size_t EllpackMemCache::GidxSizeBytes() const noexcept(true) {
  auto it = common::MakeIndexTransformIter([&](auto i) { return this->GidxSizeBytes(i); });
  using T = std::iterator_traits<decltype(it)>::value_type;
  return std::accumulate(it, it + this->Size(), static_cast<T>(0));
}

[[nodiscard]] EllpackMemCache::PagePtr EllpackMemCache::At(std::int32_t k) const {
  auto const* h_ptr = this->h_pages.at(k).get();
  auto const* d_ptr = &this->d_pages.at(k);
  return std::make_pair(h_ptr, d_ptr);
}

[[nodiscard]] EllpackMemCache::PageRef EllpackMemCache::Back() {
  auto& h_ref = this->h_pages.back();
  auto& d_ref = this->d_pages.back();
  return {h_ref, d_ref};
}

/**
 * Cache stream.
 */
class EllpackHostCacheStreamImpl {
  std::shared_ptr<EllpackMemCache> cache_;
  std::int32_t ptr_{0};

 public:
  explicit EllpackHostCacheStreamImpl(std::shared_ptr<EllpackMemCache> cache)
      : cache_{std::move(cache)} {}

  auto Share() const { return this->cache_; }

  void Seek(bst_idx_t offset_bytes) {
    std::size_t n_bytes{0};
    std::int32_t k{-1};
    for (std::size_t i = 0, n = cache_->h_pages.size(); i < n; ++i) {
      if (n_bytes == offset_bytes) {
        k = i;
        break;
      }
      n_bytes += this->cache_->SizeBytes(i);
    }
    if (offset_bytes == n_bytes && k == -1) {
      k = this->cache_->h_pages.size();  // seek end
    }
    CHECK_NE(k, -1) << "Invalid offset:" << offset_bytes;
    ptr_ = k;
  }

  [[nodiscard]] bool Write(EllpackPage const& page) {
    auto impl = page.Impl();
    auto ctx = Context{}.MakeCUDA(dh::CurrentDevice());

    this->cache_->sizes_orig.push_back(page.Impl()->MemCostBytes());
    auto orig_ptr = this->cache_->sizes_orig.size() - 1;

    CHECK_LT(orig_ptr, this->cache_->NumBatchesOrig());
    auto cache_idx = this->cache_->cache_mapping.at(orig_ptr);
    // Wrap up the previous page if this is a new page, or this is the last page.
    auto new_page = cache_idx == this->cache_->h_pages.size();
    // Last page expected from the user.
    auto last_page = (orig_ptr + 1) == this->cache_->NumBatchesOrig();

    bool const no_concat = this->cache_->NoConcat();

    auto cache_host_ratio = this->cache_->cache_host_ratio;
    CHECK_GE(cache_host_ratio, 0) << error::CacheHostRatioInvalid();
    CHECK_LE(cache_host_ratio, 1) << error::CacheHostRatioInvalid();

    // Get the size of the host cache.
    auto get_host_nbytes = [&](EllpackPageImpl const* old_impl) {
      // Special handling due to floating points.
      if (this->cache_->cache_host_ratio == 1.0) {
        return old_impl->gidx_buffer.size_bytes();
      }
      if (this->cache_->cache_host_ratio == 0.0) {
        return static_cast<std::size_t>(0);
      }
      // Calculate based on the `cache_host_ratio` parameter.
      auto n_bytes =
          std::max(static_cast<std::size_t>(old_impl->gidx_buffer.size_bytes() * cache_host_ratio),
                   std::size_t{1});
      return n_bytes;
    };
    // Finish writing a (concatenated) cache page.
    auto commit_page = [cache_host_ratio, get_host_nbytes](EllpackPageImpl const* old_impl) {
      CHECK_EQ(old_impl->gidx_buffer.Resource()->Type(), common::ResourceHandler::kCudaMalloc);
      auto new_impl = std::make_unique<EllpackPageImpl>();
      new_impl->CopyInfo(old_impl);
      // Split the cache into host cache and device cache

      // Host cache
      auto n_bytes = get_host_nbytes(old_impl);
      CHECK_LE(n_bytes, old_impl->gidx_buffer.size_bytes());
      new_impl->gidx_buffer =
          common::MakeFixedVecWithPinnedMalloc<common::CompressedByteT>(n_bytes);
      if (n_bytes > 0) {
        dh::safe_cuda(cudaMemcpyAsync(new_impl->gidx_buffer.data(), old_impl->gidx_buffer.data(),
                                      n_bytes, cudaMemcpyDefault));
      }

      // Device cache
      auto remaining = old_impl->gidx_buffer.size_bytes() - n_bytes;
      auto d_page = common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(remaining);
      if (remaining > 0) {
        dh::safe_cuda(cudaMemcpyAsync(d_page.data(), old_impl->gidx_buffer.data() + n_bytes,
                                      remaining, cudaMemcpyDefault));
      }
      CHECK_LE(new_impl->gidx_buffer.size(), old_impl->gidx_buffer.size());
      CHECK_EQ(new_impl->MemCostBytes() + d_page.size_bytes(), old_impl->MemCostBytes());
      LOG(INFO) << "Create cache page with size:"
                << common::HumanMemUnit(new_impl->MemCostBytes() + d_page.size_bytes());
      return std::make_pair(std::move(new_impl), std::move(d_page));
    };

    if (no_concat) {
      CHECK(new_page);
      auto old_impl = page.Impl();
      auto [commited, d_page] = commit_page(old_impl);

      this->cache_->offsets.push_back(old_impl->n_rows * old_impl->info.row_stride);
      this->cache_->h_pages.emplace_back(std::move(commited));
      this->cache_->d_pages.emplace_back(std::move(d_page));
      return new_page;
    }

    if (new_page) {
      if (!this->cache_->h_pages.empty()) {
        // Need to wrap up the previous page.
        // Replace the previous page (on device) with a new page on host.
        this->cache_->Back() = commit_page(this->cache_->h_pages.back().get());
      }
      // Push a new page
      auto n_bytes = this->cache_->buffer_bytes.at(this->cache_->h_pages.size());
      auto n_samples = this->cache_->buffer_rows.at(this->cache_->h_pages.size());
      auto new_impl = std::make_unique<EllpackPageImpl>(&ctx, impl->CutsShared(), impl->IsDense(),
                                                        impl->info.row_stride, n_samples);
      new_impl->SetBaseRowId(impl->base_rowid);
      new_impl->SetNumSymbols(impl->NumSymbols());
      new_impl->gidx_buffer =
          common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(&ctx, n_bytes, 0);
      auto offset = new_impl->Copy(&ctx, impl, 0);

      this->cache_->offsets.push_back(offset);

      // Make sure we can always access the back of the vectors
      this->cache_->h_pages.emplace_back(std::move(new_impl));
      this->cache_->d_pages.emplace_back();
    } else {
      // Concatenate into the device pages even though `d_pages` is used. We split the
      // page at the commit stage.
      CHECK(!this->cache_->h_pages.empty());
      CHECK_EQ(cache_idx, this->cache_->h_pages.size() - 1);
      auto& new_impl = this->cache_->h_pages.back();
      auto offset = new_impl->Copy(&ctx, impl, this->cache_->offsets.back());
      this->cache_->offsets.back() += offset;
    }

    // No need to copy if it's already in device.
    if (last_page) {
      this->cache_->Back() = commit_page(this->cache_->h_pages.back().get());
    }

    CHECK_EQ(this->cache_->h_pages.size(), this->cache_->d_pages.size());
    return new_page;
  }

  void Read(EllpackPage* out, bool prefetch_copy) const {
    CHECK_EQ(this->cache_->h_pages.size(), this->cache_->d_pages.size());
    auto [h_page, d_page] = this->cache_->At(this->ptr_);
    // Skip copy if the full page is on device
    bool on_device = h_page->gidx_buffer.empty() && !d_page->empty();
    auto ctx = Context{}.MakeCUDA(dh::CurrentDevice());
    auto out_impl = out->Impl();
    if (on_device) {
      CHECK(h_page->gidx_buffer.empty());
      auto d_res = d_page->Resource();
      out_impl->gidx_buffer = common::RefResourceView<common::CompressedByteT>{
          d_res->DataAs<common::CompressedByteT>(), d_page->size(), d_res};
      CHECK(out_impl->d_gidx_buffer.empty());
    } else if (prefetch_copy) {
      auto n_bytes = this->cache_->GidxSizeBytes(this->ptr_);
      out_impl->gidx_buffer = common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(n_bytes);
      if (!h_page->gidx_buffer.empty()) {
        dh::safe_cuda(cudaMemcpyAsync(out_impl->gidx_buffer.data(), h_page->gidx_buffer.data(),
                                      h_page->gidx_buffer.size_bytes(), cudaMemcpyDefault,
                                      ctx.CUDACtx()->Stream()));
      }
      if (!d_page->empty()) {
        auto beg = out_impl->gidx_buffer.data() + h_page->gidx_buffer.size();
        dh::safe_cuda(cudaMemcpyAsync(beg, d_page->data(), d_page->size_bytes(), cudaMemcpyDefault,
                                      ctx.CUDACtx()->Stream()));
      }
    } else {
      auto h_res = h_page->gidx_buffer.Resource();
      CHECK(h_res->DataAs<common::CompressedByteT>() == h_page->gidx_buffer.data());
      out_impl->gidx_buffer = common::RefResourceView<common::CompressedByteT>{
          h_res->DataAs<common::CompressedByteT>(), h_page->gidx_buffer.size(), h_res};
      CHECK(out_impl->d_gidx_buffer.empty());
      if (!d_page->empty()) {
        out_impl->d_gidx_buffer = common::RefResourceView<common::CompressedByteT const>{
            d_page->data(), d_page->size(), d_page->Resource()};
      }
    }

    out_impl->CopyInfo(h_page);
  }
};

/**
 * EllpackHostCacheStream
 */
EllpackHostCacheStream::EllpackHostCacheStream(std::shared_ptr<EllpackMemCache> cache)
    : p_impl_{std::make_unique<EllpackHostCacheStreamImpl>(std::move(cache))} {}

EllpackHostCacheStream::~EllpackHostCacheStream() = default;

std::shared_ptr<EllpackMemCache const> EllpackHostCacheStream::Share() const {
  return p_impl_->Share();
}

void EllpackHostCacheStream::Seek(bst_idx_t offset_bytes) { this->p_impl_->Seek(offset_bytes); }

void EllpackHostCacheStream::Read(EllpackPage* page, bool prefetch_copy) const {
  this->p_impl_->Read(page, prefetch_copy);
}

[[nodiscard]] bool EllpackHostCacheStream::Write(EllpackPage const& page) {
  return this->p_impl_->Write(page);
}

/**
 * EllpackCacheStreamPolicy
 */
template <typename S, template <typename> typename F>
[[nodiscard]] std::unique_ptr<typename EllpackCacheStreamPolicy<S, F>::WriterT>
EllpackCacheStreamPolicy<S, F>::CreateWriter(StringView, std::uint32_t iter) {
  if (!this->p_cache_) {
    CHECK(!detail::HostRatioIsAuto(this->CacheInfo().cache_host_ratio));
    CHECK_GE(this->CacheInfo().cache_host_ratio, 0.0);
    CHECK_LE(this->CacheInfo().cache_host_ratio, 1.0);
    constexpr std::int32_t kMaxGpuExtMemWorkers = 4;
    this->p_cache_ = std::make_unique<EllpackMemCache>(this->CacheInfo(), kMaxGpuExtMemWorkers);
  }
  auto fo = std::make_unique<EllpackHostCacheStream>(this->p_cache_);
  if (iter == 0) {
    CHECK(this->p_cache_->Empty());
  } else {
    fo->Seek(this->p_cache_->SizeBytes());
  }
  return fo;
}

template <typename S, template <typename> typename F>
[[nodiscard]] std::unique_ptr<typename EllpackCacheStreamPolicy<S, F>::ReaderT>
EllpackCacheStreamPolicy<S, F>::CreateReader(StringView, bst_idx_t offset, bst_idx_t) const {
  auto fi = std::make_unique<ReaderT>(this->p_cache_);
  fi->Seek(offset);
  return fi;
}

// Instantiation
template std::unique_ptr<
    typename EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::WriterT>
EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::CreateWriter(StringView name,
                                                                         std::uint32_t iter);

template std::unique_ptr<
    typename EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::ReaderT>
EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::CreateReader(StringView name,
                                                                         bst_idx_t offset,
                                                                         bst_idx_t length) const;

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

void CalcCacheMapping(Context const* ctx, bool is_dense,
                      std::shared_ptr<common::HistogramCuts const> cuts,
                      std::int64_t min_cache_page_bytes, ExternalDataInfo const& ext_info,
                      bool is_validation, EllpackCacheInfo* cinfo) {
  CHECK(cinfo->param.Initialized()) << "Need to initialize scalar fields first.";
  auto ell_info = CalcNumSymbols(ctx, ext_info.row_stride, is_dense, cuts);

  /**
   * Configure the cache
   */
  // The total size of the cache.
  std::size_t n_cache_bytes = 0;
  for (std::size_t i = 0; i < ext_info.n_batches; ++i) {
    auto n_samples = ext_info.base_rowids.at(i + 1) - ext_info.base_rowids[i];
    auto n_bytes = common::CompressedBufferWriter::CalculateBufferSize(
        ext_info.row_stride * n_samples, ell_info.n_symbols);
    n_cache_bytes += n_bytes;
  }
  std::tie(cinfo->cache_host_ratio, min_cache_page_bytes) = detail::DftPageSizeHostRatio(
      n_cache_bytes, is_validation, cinfo->cache_host_ratio, min_cache_page_bytes);

  /**
   * Calculate the cache buffer size
   */
  std::vector<std::size_t> cache_bytes;
  std::vector<std::size_t> cache_mapping(ext_info.n_batches, 0);
  std::vector<std::size_t> cache_rows;

  for (std::size_t i = 0; i < ext_info.n_batches; ++i) {
    auto n_samples = ext_info.base_rowids[i+1] - ext_info.base_rowids[i];
    auto n_bytes = common::CompressedBufferWriter::CalculateBufferSize(
        ext_info.row_stride * n_samples, ell_info.n_symbols);

    if (cache_bytes.empty()) {
      // Push the first page
      cache_bytes.push_back(n_bytes);
      cache_rows.push_back(n_samples);
    } else if (static_cast<decltype(min_cache_page_bytes)>(cache_bytes.back()) <
               min_cache_page_bytes) {
      // Concatenate to the previous page
      cache_bytes.back() += n_bytes;
      cache_rows.back() += n_samples;
    } else {
      // Push a new page
      cache_bytes.push_back(n_bytes);
      cache_rows.push_back(n_samples);
    }
    cache_mapping[i] = cache_bytes.size() - 1;
  }

  cinfo->cache_mapping = std::move(cache_mapping);
  cinfo->buffer_bytes = std::move(cache_bytes);
  cinfo->buffer_rows = std::move(cache_rows);

  // Directly store in device if there's only one batch.
  if (cinfo->NumBatchesCc() == 1) {
    cinfo->cache_host_ratio = 0.0;
  }

  LOG(INFO) << "`cache_host_ratio`=" << cinfo->cache_host_ratio
            << " `min_cache_page_bytes`=" << min_cache_page_bytes;
}

/**
 * EllpackPageSourceImpl
 */
template <typename F>
void EllpackPageSourceImpl<F>::Fetch() {
  curt::SetDevice(this->Device().ordinal);
  if (!this->ReadCache()) {
    if (this->Iter() != 0 && !this->sync_) {
      // source is initialized to be the 0th page during construction, so when count_ is 0
      // there's no need to increment the source.
      ++(*this->source_);
    }
    // This is not read from cache so we still need it to be synced with sparse page source.
    CHECK_EQ(this->Iter(), this->source_->Iter());
    auto const& csr = this->source_->Page();
    this->page_.reset(new EllpackPage{});
    auto* impl = this->page_->Impl();
    Context ctx = Context{}.MakeCUDA(this->Device().ordinal);
    if (this->GetCuts()->HasCategorical()) {
      CHECK(!this->feature_types_.empty());
    }
    *impl = EllpackPageImpl{&ctx, this->GetCuts(), *csr, is_dense_, row_stride_, feature_types_};
    this->page_->SetBaseRowId(csr->base_rowid);
    LOG(INFO) << "Generated an Ellpack page with size: "
              << common::HumanMemUnit(impl->MemCostBytes())
              << " from a SparsePage with size:" << common::HumanMemUnit(csr->MemCostBytes());
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
  curt::SetDevice(this->Device().ordinal);
  if (!this->ReadCache()) {
    auto iter = this->source_->Iter();
    CHECK_EQ(this->Iter(), iter);
    cuda_impl::Dispatch(proxy_, [this](auto const& value) {
      CHECK(this->proxy_->Ctx()->IsCUDA()) << "All batches must use the same device type.";
      proxy_->Info().feature_types.SetDevice(dh::GetDevice(this->ctx_));
      auto d_feature_types = proxy_->Info().feature_types.ConstDeviceSpan();
      auto n_samples = value.NumRows();
      if (this->GetCuts()->HasCategorical()) {
        CHECK(!d_feature_types.empty());
      }
      dh::device_vector<size_t> row_counts(n_samples + 1, 0);
      common::Span<size_t> row_counts_span(row_counts.data().get(), row_counts.size());
      bst_idx_t row_stride = GetRowCounts(this->ctx_, value, row_counts_span,
                                          dh::GetDevice(this->ctx_), this->missing_);
      CHECK_LE(row_stride, this->ext_info_.row_stride);
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
    LOG(INFO) << "Generated an Ellpack page with size: "
              << common::HumanMemUnit(this->page_->Impl()->MemCostBytes())
              << " from an batch with estimated size: "
              << cuda_impl::Dispatch<false>(proxy_, [](auto const& adapter) {
                   return common::HumanMemUnit(adapter->SizeBytes());
                 });
    this->page_->SetBaseRowId(this->ext_info_.base_rowids.at(iter));
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
