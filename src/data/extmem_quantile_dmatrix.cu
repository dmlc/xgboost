/**
 * Copyright 2024, XGBoost Contributors
 *
 * The @ref ExtMemQuantileDMatrix for GPU prefetches 2 pages by default and can optionally
 * cache page in the device memory for the validation DMatrix. In addition, it can
 * concatenate user-provded pages to form larger @ref EllpackPage to avoid small GPU
 * kernels.
 *
 * Given 1 training DMatrix and 1 validation DMatrix, with 2 pages from the validation set
 * cached in device memory, we can have at most 6 pages in the device memory. 2 from
 * prefetched training DMatrix, 2 from prefetched validation DMatrix, and 2 in the device
 * cache. If set the minimum @ref EllpackPage to 12GB in a 96GB GPU, 6 pages have 72GB
 * size in total. Without accounting for memory fragmentation, this should be very close
 * the upper boundary.
 */
#include <memory>   // for shared_ptr
#include <variant>  // for visit, get_if

#include "../common/cuda_rt_utils.h"  // for xgboost_NVTX_FN_RANGE
#include "batch_utils.h"              // for CheckParam, RegenGHist
#include "ellpack_page.cuh"           // for EllpackPage
#include "extmem_quantile_dmatrix.h"
#include "proxy_dmatrix.h"    // for DataIterProxy
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"     // for BatchParam
#include "batch_utils.h"      // for AutoCachePageBytes

namespace xgboost::data {
[[nodiscard]] std::int64_t DftMinCachePageBytes(std::int64_t min_cache_page_bytes) {
  // Set to 0 if it should match the user input size.
  if (::xgboost::cuda_impl::AutoCachePageBytes() == min_cache_page_bytes) {
    double n_total_bytes = curt::TotalMemory();
    min_cache_page_bytes = n_total_bytes * xgboost::cuda_impl::CachePageRatio();
  }
  return min_cache_page_bytes;
}

void ExtMemQuantileDMatrix::InitFromCUDA(
    Context const *ctx,
    std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> iter,
    DMatrixHandle proxy_handle, BatchParam const &p, std::shared_ptr<DMatrix> ref,
    std::int64_t max_quantile_blocks, ExtMemConfig const &config) {
  xgboost_NVTX_FN_RANGE();

  // A handle passed to external iterator.
  auto proxy = MakeProxy(proxy_handle);
  CHECK(proxy);

  /**
   * Generate quantiles
   */
  auto cuts = std::make_shared<common::HistogramCuts>();
  ExternalDataInfo ext_info;
  cuda_impl::MakeSketches(ctx, iter.get(), proxy, ref, p, config.missing, cuts, this->info_,
                          max_quantile_blocks, &ext_info);
  ext_info.SetInfo(ctx, &this->info_);

  /**
   * Calculate cache info
   */
  // Prefer device storage for validation dataset since we can't hide it's data load
  // overhead with inference. But the training procedures can confortably overlap with the
  // data transfer.
  auto cinfo = EllpackCacheInfo{p, (ref != nullptr), config.max_num_device_pages, config.missing};
  CalcCacheMapping(ctx, this->info_.IsDense(), cuts,
                   DftMinCachePageBytes(config.min_cache_page_bytes), ext_info, &cinfo);
  CHECK_EQ(cinfo.cache_mapping.size(), ext_info.n_batches);
  auto n_batches = cinfo.buffer_rows.size();  // The number of batches after page concatenation.
  LOG(INFO) << "Number of batches after concatenation:" << n_batches;

  /**
   * Generate gradient index
   */
  auto id = MakeCache(this, ".ellpack.page", this->on_host_, cache_prefix_, &cache_info_);
  if (on_host_ && std::get_if<EllpackHostPtr>(&ellpack_page_source_) == nullptr) {
    ellpack_page_source_.emplace<EllpackHostPtr>(nullptr);
  }

  std::visit(
      [&](auto &&ptr) {
        using SourceT = typename std::remove_reference_t<decltype(ptr)>::element_type;
        ptr = std::make_shared<SourceT>(ctx, &this->info_, ext_info, cache_info_.at(id), cuts, iter,
                                        proxy, cinfo);
      },
      ellpack_page_source_);

  /**
   * Force initialize the cache and do some sanity checks along the way
   */
  bst_idx_t batch_cnt = 0, k = 0;
  bst_idx_t n_total_samples = 0;
  for (auto const &page : this->GetEllpackPageImpl()) {
    n_total_samples += page.Size();
    CHECK_EQ(page.Impl()->base_rowid, ext_info.base_rowids[k]);
    CHECK_EQ(page.Impl()->info.row_stride, ext_info.row_stride);
    ++k, ++batch_cnt;
  }
  CHECK_EQ(batch_cnt, ext_info.n_batches);
  CHECK_EQ(n_total_samples, ext_info.accumulated_rows);

  if (this->on_host_) {
    CHECK_EQ(this->cache_info_.at(id)->Size(), n_batches);
  } else {
    CHECK_EQ(this->cache_info_.at(id)->Size(), ext_info.n_batches);
  }
  this->n_batches_ = this->cache_info_.at(id)->Size();
  if (cuts->HasCategorical()) {
    CHECK(!this->info_.feature_types.Empty());
  }
  CHECK_EQ(cuts->HasCategorical(), this->info_.HasCategorical());
}

[[nodiscard]] BatchSet<EllpackPage> ExtMemQuantileDMatrix::GetEllpackPageImpl() {
  auto batch_set =
      std::visit([this](auto &&ptr) { return BatchSet{BatchIterator<EllpackPage>{ptr}}; },
                 this->ellpack_page_source_);
  return batch_set;
}

BatchSet<EllpackPage> ExtMemQuantileDMatrix::GetEllpackBatches(Context const *,
                                                               const BatchParam &param) {
  if (param.Initialized()) {
    detail::CheckParam(this->batch_, param);
    CHECK(!detail::RegenGHist(param, batch_)) << error::InconsistentMaxBin();
  }

  std::visit(
      [this, param](auto &&ptr) {
        CHECK(ptr)
            << "The `ExtMemQuantileDMatrix` is initialized using CPU data, cannot be used for GPU.";
        ptr->Reset(param);
      },
      this->ellpack_page_source_);

  return this->GetEllpackPageImpl();
}
}  // namespace xgboost::data
