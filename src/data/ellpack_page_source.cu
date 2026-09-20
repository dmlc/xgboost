/**
 * Copyright 2019-2026, XGBoost contributors
 */
#include <algorithm>  // for fill, max
#include <cstddef>    // for size_t
#include <cstdint>    // for int8_t, uint64_t, uint32_t
#include <memory>     // for shared_ptr, make_unique
#include <numeric>    // for accumulate
#include <utility>    // for move

#include "../common/common.h"               // for HumanMemUnit, safe_cuda
#include "../common/cuda_context.cuh"       // for CUDAContext
#include "../common/cuda_rt_utils.h"        // for SetDevice
#include "../common/device_helpers.cuh"     // for CurrentDevice
#include "../common/numa_topo.h"            // for NumaMemCanCross, GetNumaMemBind
#include "../common/ref_resource_view.cuh"  // for MakeFixedVecWithCudaMalloc
#include "../common/resource.cuh"           // for PrivateCudaMmapConstStream
#include "../common/transform_iterator.h"   // for MakeIndexTransformIter
#include "batch_utils.h"                    // for HostRatioIsAuto
#include "ellpack_page.cuh"                 // for EllpackPageImpl
#include "ellpack_page.h"                   // for EllpackPage
#include "ellpack_page_source.h"
#include "proxy_dmatrix.cuh"  // for DispatchAny
#include "xgboost/base.h"     // for bst_idx_t

namespace xgboost::data {
/**
 * Cache
 */
EllpackMemCache::EllpackMemCache(EllpackCacheInfo cinfo)
    : cache_mapping{std::move(cinfo.cache_mapping)},
      buffer_bytes{std::move(cinfo.buffer_bytes)},
      buffer_rows{std::move(cinfo.buffer_rows)},
      cache_host_ratio{cinfo.cache_host_ratio} {
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
  return std::make_tuple(h_ptr, d_ptr);
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

  [[nodiscard]] bool Write(Context const* ctx, EllpackPage const& page) {
    auto impl = page.Impl();

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
    auto commit_page = [&](EllpackPageImpl const* old_impl) {
      CHECK_EQ(old_impl->gidx_buffer.Resource()->Type(), common::ResourceHandler::kCudaMalloc);
      auto new_impl = std::make_unique<EllpackPageImpl>();
      new_impl->CopyInfo(old_impl);

      // Split the cache into host and device cache.
      auto n_bytes = get_host_nbytes(old_impl);
      CHECK_LE(n_bytes, old_impl->gidx_buffer.size_bytes());

      // Host cache
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
      return std::make_tuple(std::move(new_impl), std::move(d_page));
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
      auto new_impl = std::make_unique<EllpackPageImpl>(ctx, impl->CutsShared(), impl->IsDense(),
                                                        impl->info.row_stride, n_samples);
      new_impl->SetBaseRowId(impl->base_rowid);
      new_impl->SetNumSymbols(impl->NumSymbols());
      new_impl->gidx_buffer =
          common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(ctx, n_bytes, 0);
      auto offset = new_impl->Copy(ctx, impl, 0);

      this->cache_->offsets.push_back(offset);

      // Make sure we can always access the back of the vectors
      this->cache_->h_pages.emplace_back(std::move(new_impl));
      this->cache_->d_pages.emplace_back();
    } else {
      // Concatenate on device in `h_pages`. We split the page at the commit stage.
      CHECK(!this->cache_->h_pages.empty());
      CHECK_EQ(cache_idx, this->cache_->h_pages.size() - 1);
      auto& new_impl = this->cache_->h_pages.back();
      auto offset = new_impl->Copy(ctx, impl, this->cache_->offsets.back());
      this->cache_->offsets.back() += offset;
    }

    // No need to copy if it's already in device.
    if (last_page) {
      this->cache_->Back() = commit_page(this->cache_->h_pages.back().get());
    }

    CHECK_EQ(this->cache_->h_pages.size(), this->cache_->d_pages.size());
    return new_page;
  }

  void Read(Context const* ctx, EllpackPage* out, bool prefetch_copy) const {
    CHECK_EQ(this->cache_->h_pages.size(), this->cache_->d_pages.size());
    auto [h_page, d_page] = this->cache_->At(this->ptr_);
    // Skip copy if the full page is on device
    bool on_device = h_page->gidx_buffer.empty() && !d_page->empty();

    auto out_impl = out->Impl();
    LOG(DEBUG) << "On device: " << on_device << ", prefetch copy:" << prefetch_copy;
    if (on_device) {
      CHECK(h_page->gidx_buffer.empty());
      auto d_res = d_page->Resource();
      out_impl->gidx_buffer = common::RefResourceView<common::CompressedByteT>{
          d_res->DataAs<common::CompressedByteT>(), d_page->size(), d_res};
      CHECK(out_impl->d_gidx_buffer.empty());
    } else if (prefetch_copy) {
      // Copy the data in the same order as written
      // Host cache
      auto n_bytes = this->cache_->GidxSizeBytes(this->ptr_);
      out_impl->gidx_buffer = common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(n_bytes);
      if (!h_page->gidx_buffer.empty()) {
        dh::safe_cuda(cudaMemcpyAsync(out_impl->gidx_buffer.data(), h_page->gidx_buffer.data(),
                                      h_page->gidx_buffer.size_bytes(), cudaMemcpyDefault,
                                      ctx->CUDACtx()->Stream()));
      }
      // Device cache
      if (!d_page->empty()) {
        auto out = out_impl->gidx_buffer.ToSpan().subspan(h_page->gidx_buffer.size_bytes());
        CHECK_EQ(out.size_bytes(), d_page->size_bytes());
        dh::safe_cuda(cudaMemcpyAsync(out.data(), d_page->data(), d_page->size_bytes(),
                                      cudaMemcpyDefault, ctx->CUDACtx()->Stream()));
      }
    } else {
      // Direct access
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

void EllpackHostCacheStream::Read(Context const* ctx, EllpackPage* page, bool prefetch_copy) const {
  this->p_impl_->Read(ctx, page, prefetch_copy);
}

[[nodiscard]] bool EllpackHostCacheStream::Write(Context const* ctx, EllpackPage const& page) {
  return this->p_impl_->Write(ctx, page);
}

/**
 * EllpackFormatPolicy
 */
template <typename S>
void EllpackFormatPolicy<S>::DestroyPage(std::shared_ptr<S>* page) const {
  if (page && ctx_) {
    ctx_->CUDACtx()->Stream().Sync();
  }
  page->reset();
}

template void EllpackFormatPolicy<EllpackPage>::DestroyPage(
    std::shared_ptr<EllpackPage>* page) const;

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
    this->p_cache_ = std::make_unique<EllpackMemCache>(this->CacheInfo());
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
  // Byte offsets for input batch boundaries, alongside ext_info.base_rowids.
  std::vector<std::size_t> byte_ptr(ext_info.n_batches + 1, 0);
  for (std::size_t i = 0; i < ext_info.n_batches; ++i) {
    auto n_samples = ext_info.base_rowids.at(i + 1) - ext_info.base_rowids[i];
    auto n_bytes = common::CompressedBufferWriter::CalculateBufferSize(
        ext_info.row_stride * n_samples, ell_info.n_symbols);
    byte_ptr[i + 1] = byte_ptr[i] + n_bytes;
  }
  std::tie(cinfo->cache_host_ratio, min_cache_page_bytes) = detail::DftPageSizeHostRatio(
      byte_ptr.back(), is_validation, cinfo->cache_host_ratio, min_cache_page_bytes);

  /**
   * Calculate the cache buffer size
   */
  // Keep the page count selected by the minimum size, but avoid a small final page that
  // cannot hide the transfer of the next page during histogram construction. (balancing)
  //
  // n_batches_cc is the same as the `EllpackCacheInfo::NumBatchesCc`.
  std::size_t n_batches_cc = 0, page_bytes = 0;
  for (std::size_t i = 0; i < ext_info.n_batches; ++i) {
    if (n_batches_cc == 0 || static_cast<std::int64_t>(page_bytes) >= min_cache_page_bytes) {
      ++n_batches_cc;
      page_bytes = 0;
    }
    page_bytes += byte_ptr[i + 1] - byte_ptr[i];
  }
  std::vector<std::size_t> cache_bytes(n_batches_cc, 0);
  // Mapping from the user inputs to the internal batches.
  std::vector<std::size_t> cache_mapping(ext_info.n_batches, 0);
  std::vector<std::size_t> cache_rows(n_batches_cc, 0);
  for (std::size_t p = 0, input_batch_begin = 0; p < n_batches_cc; ++p) {
    auto remaining_pages = n_batches_cc - p;
    // Average size of the remaining pages
    auto target =
        common::DivRoundUp(byte_ptr.back() - byte_ptr[input_batch_begin], remaining_pages);
    // Difference between the proposed page size and its target size.
    auto distance = [&](std::size_t input_batch_end) {
      auto n_bytes = byte_ptr[input_batch_end] - byte_ptr[input_batch_begin];
      return n_bytes > target ? n_bytes - target : target - n_bytes;
    };
    // Leave at least one input batch for every subsequent page.
    auto min_input_batches_to_reserve = (remaining_pages - 1);
    auto input_batch_end_limit = ext_info.n_batches - min_input_batches_to_reserve;
    auto input_batch_end = input_batch_begin + 1;
    // Grow toward the target, preferring the larger page on a tie.
    while (input_batch_end < input_batch_end_limit &&
           distance(input_batch_end + 1) <= distance(input_batch_end)) {
      ++input_batch_end;
    }
    // Concatenate the batches between begin and end
    cache_bytes[p] = byte_ptr[input_batch_end] - byte_ptr[input_batch_begin];
    cache_rows[p] = ext_info.base_rowids[input_batch_end] - ext_info.base_rowids[input_batch_begin];
    std::fill(cache_mapping.begin() + input_batch_begin, cache_mapping.begin() + input_batch_end,
              p);
    input_batch_begin = input_batch_end;
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
    this->DestroyPage(&this->page_);
    this->page_.reset(new EllpackPage{});
    auto* impl = this->page_->Impl();
    if (this->GetCuts()->HasCategorical()) {
      CHECK(!this->feature_types_.empty());
    }
    *impl =
        EllpackPageImpl{this->Ctx(), this->GetCuts(), *csr, is_dense_, row_stride_, feature_types_};
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
    cuda_impl::DispatchAny(proxy_, [this](auto const& value) {
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
      this->DestroyPage(&this->page_);
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
    LOG(DEBUG) << "Generated an Ellpack page with size: "
               << common::HumanMemUnit(this->page_->Impl()->MemCostBytes())
               << " from an batch with estimated size: "
               << cuda_impl::DispatchAny<false>(proxy_, [](auto const& adapter) {
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

namespace detail {
void EllpackFormatCheckNuma(StringView msg) {
#if defined(__linux__)
  bool can_cross = common::NumaMemCanCross();
  std::uint32_t numa = 0;
  auto incorrect = [&numa] {
    std::uint32_t cpu = 0;
    return common::GetCpuNuma(&cpu, &numa) && static_cast<std::int32_t>(numa) != curt::GetNumaId();
  };

  if (can_cross && !common::GetNumaMemBind()) {
    LOG(WARNING) << "Running on a NUMA system without membind." << msg;
  } else if (can_cross && incorrect()) {
    LOG(WARNING) << "Incorrect NUMA CPU bind, CPU node:" << numa
                 << ", GPU node:" << curt::GetNumaId() << "." << msg;
  }
#else
  (void)msg;
#endif
}
}  // namespace detail
}  // namespace xgboost::data
