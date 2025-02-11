/**
 * Copyright 2019-2025, XGBoost contributors
 */
#include <algorithm>  // for count_if
#include <cstddef>    // for size_t
#include <cstdint>    // for int8_t, uint64_t, uint32_t
#include <memory>     // for shared_ptr, make_unique, make_shared
#include <numeric>    // for accumulate
#include <utility>    // for move

#include "../common/common.h"               // for HumanMemUnit, safe_cuda
#include "../common/cuda_rt_utils.h"        // for SetDevice
#include "../common/device_helpers.cuh"     // for CUDAStreamView, DefaultStream
#include "../common/ref_resource_view.cuh"  // for MakeFixedVecWithCudaMalloc
#include "../common/resource.cuh"           // for PrivateCudaMmapConstStream
#include "../common/transform_iterator.h"   // for MakeIndexTransformIter
#include "ellpack_page.cuh"                 // for EllpackPageImpl
#include "ellpack_page.h"                   // for EllpackPage
#include "ellpack_page_source.h"
#include "proxy_dmatrix.cuh"  // for Dispatch
#include "xgboost/base.h"     // for bst_idx_t

namespace xgboost::data {
namespace {
[[nodiscard]] bool IsDevicePage(EllpackPageImpl const* page) {
  switch (page->gidx_buffer.Resource()->Type()) {
    case common::ResourceHandler::kCudaMalloc:
    case common::ResourceHandler::kCudaGrowOnly: {
      return true;
    }
    case common::ResourceHandler::kCudaHostCache:
    case common::ResourceHandler::kCudaMmap:
    case common::ResourceHandler::kMmap:
    case common::ResourceHandler::kMalloc:
      return false;
  }
  LOG(FATAL) << "Unreachable";
  return false;
}
}  // anonymous namespace

/**
 * Cache
 */
EllpackMemCache::EllpackMemCache(EllpackCacheInfo cinfo)
    : cache_mapping{std::move(cinfo.cache_mapping)},
      buffer_bytes{std::move(cinfo.buffer_bytes)},
      buffer_rows{std::move(cinfo.buffer_rows)},
      prefer_device{cinfo.prefer_device},
      max_num_device_pages{cinfo.max_num_device_pages} {
  CHECK_EQ(buffer_bytes.size(), buffer_rows.size());
}

EllpackMemCache::~EllpackMemCache() = default;

[[nodiscard]] std::size_t EllpackMemCache::SizeBytes() const {
  auto it = common::MakeIndexTransformIter([&](auto i) { return pages.at(i)->MemCostBytes(); });
  using T = std::iterator_traits<decltype(it)>::value_type;
  return std::accumulate(it, it + pages.size(), static_cast<T>(0));
}

[[nodiscard]] EllpackPageImpl const* EllpackMemCache::At(std::int32_t k) const {
  return this->pages.at(k).get();
}

[[nodiscard]] std::int64_t EllpackMemCache::NumDevicePages() const {
  return std::count_if(this->pages.cbegin(), this->pages.cend(),
                       [](auto const& page) { return IsDevicePage(page.get()); });
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

  auto Share() { return cache_; }

  void Seek(bst_idx_t offset_bytes) {
    std::size_t n_bytes{0};
    std::int32_t k{-1};
    for (std::size_t i = 0, n = cache_->pages.size(); i < n; ++i) {
      if (n_bytes == offset_bytes) {
        k = i;
        break;
      }
      n_bytes += cache_->pages[i]->MemCostBytes();
    }
    if (offset_bytes == n_bytes && k == -1) {
      k = this->cache_->pages.size();  // seek end
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
    auto new_page = cache_idx == this->cache_->pages.size();

    auto last_page = (orig_ptr + 1) == this->cache_->NumBatchesOrig();
    // No page concatenation is performed. If there's page concatenation, then the number
    // of pages in the cache must be smaller than the input number of pages.
    bool no_concat = this->cache_->NumBatchesOrig() == this->cache_->buffer_rows.size();
    // Whether the page should be cached in device. If true, then we don't need to make a
    // copy during write since the temporary page is already in device when page
    // concatenation is enabled.
    bool to_device = this->cache_->prefer_device &&
                     this->cache_->NumDevicePages() < this->cache_->max_num_device_pages;

    auto commit_page = [&ctx](EllpackPageImpl const* old_impl) {
      CHECK_EQ(old_impl->gidx_buffer.Resource()->Type(), common::ResourceHandler::kCudaMalloc);
      auto new_impl = std::make_unique<EllpackPageImpl>();
      new_impl->CopyInfo(old_impl);
      new_impl->gidx_buffer = common::MakeFixedVecWithPinnedMalloc<common::CompressedByteT>(
          old_impl->gidx_buffer.size());
      dh::safe_cuda(cudaMemcpyAsync(new_impl->gidx_buffer.data(), old_impl->gidx_buffer.data(),
                                    old_impl->gidx_buffer.size_bytes(), cudaMemcpyDefault));
      LOG(INFO) << "Create cache page with size:" << common::HumanMemUnit(new_impl->MemCostBytes());
      return new_impl;
    };
    if (no_concat) {
      // Avoid a device->device->host copy.
      CHECK(new_page);
      auto new_impl = std::make_unique<EllpackPageImpl>();
      new_impl->CopyInfo(page.Impl());

      if (to_device) {
        // Copy to device
        new_impl->gidx_buffer = common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(
            page.Impl()->gidx_buffer.size());
      } else {
        // Copy to host
        new_impl->gidx_buffer = common::MakeFixedVecWithPinnedMalloc<common::CompressedByteT>(
            page.Impl()->gidx_buffer.size());
      }
      dh::safe_cuda(cudaMemcpyAsync(new_impl->gidx_buffer.data(), page.Impl()->gidx_buffer.data(),
                                    page.Impl()->gidx_buffer.size_bytes(), cudaMemcpyDefault));

      this->cache_->offsets.push_back(new_impl->n_rows * new_impl->info.row_stride);
      this->cache_->pages.push_back(std::move(new_impl));
      return new_page;
    }

    if (new_page) {
      // No need to copy if it's already in device.
      if (!this->cache_->pages.empty() && !to_device) {
        // Need to wrap up the previous page.
        auto commited = commit_page(this->cache_->pages.back().get());
        // Replace the previous page with a new page.
        this->cache_->pages.back() = std::move(commited);
      }
      // Push a new page
      auto n_bytes = this->cache_->buffer_bytes.at(this->cache_->pages.size());
      auto n_samples = this->cache_->buffer_rows.at(this->cache_->pages.size());
      auto new_impl = std::make_unique<EllpackPageImpl>(&ctx, impl->CutsShared(), impl->IsDense(),
                                                        impl->info.row_stride, n_samples);
      new_impl->SetBaseRowId(impl->base_rowid);
      new_impl->SetNumSymbols(impl->NumSymbols());
      new_impl->gidx_buffer =
          common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(&ctx, n_bytes, 0);
      auto offset = new_impl->Copy(&ctx, impl, 0);

      this->cache_->offsets.push_back(offset);
      this->cache_->pages.push_back(std::move(new_impl));
    } else {
      CHECK(!this->cache_->pages.empty());
      CHECK_EQ(cache_idx, this->cache_->pages.size() - 1);
      auto& new_impl = this->cache_->pages.back();
      auto offset = new_impl->Copy(&ctx, impl, this->cache_->offsets.back());
      this->cache_->offsets.back() += offset;
      // No need to copy if it's already in device.
      if (last_page && !to_device) {
        auto commited = commit_page(this->cache_->pages.back().get());
        this->cache_->pages.back() = std::move(commited);
      }
    }

    return new_page;
  }

  void Read(EllpackPage* out, bool prefetch_copy) const {
    auto page = this->cache_->At(this->ptr_);
    if (IsDevicePage(page)) {
      // Page is already in the device memory, no need to copy.
      prefetch_copy = false;
    }
    auto out_impl = out->Impl();
    if (prefetch_copy) {
      out_impl->gidx_buffer =
          common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(page->gidx_buffer.size());
      dh::safe_cuda(cudaMemcpyAsync(out_impl->gidx_buffer.data(), page->gidx_buffer.data(),
                                    page->gidx_buffer.size_bytes(), cudaMemcpyDefault));
    } else {
      auto res = page->gidx_buffer.Resource();
      out_impl->gidx_buffer = common::RefResourceView<common::CompressedByteT>{
          res->DataAs<common::CompressedByteT>(), page->gidx_buffer.size(), res};
    }

    out_impl->CopyInfo(page);
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
                      EllpackCacheInfo* cinfo) {
  CHECK(cinfo->param.Initialized()) << "Need to initialize scalar fields first.";
  auto ell_info = CalcNumSymbols(ctx, ext_info.row_stride, is_dense, cuts);
  std::vector<std::size_t> cache_bytes;
  std::vector<std::size_t> cache_mapping(ext_info.n_batches, 0);
  std::vector<std::size_t> cache_rows;

  for (std::size_t i = 0; i < ext_info.n_batches; ++i) {
    auto n_samples = ext_info.base_rowids.at(i + 1) - ext_info.base_rowids[i];
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
