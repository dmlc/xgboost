/**
 * Copyright 2024, XGBoost Contributors
 */
#include "extmem_quantile_dmatrix.h"

#include <memory>  // for shared_ptr
#include <string>  // for string
#include <vector>  // for vector

#include "../tree/param.h"          // FIXME(jiamingy): Find a better way to share this parameter.
#include "batch_utils.h"            // for CheckParam, RegenGHist
#include "proxy_dmatrix.h"          // for DataIterProxy
#include "quantile_dmatrix.h"       // for GetDataShape, MakeSketches
#include "simple_batch_iterator.h"  // for SimpleBatchIteratorImpl
#include "sparse_page_source.h"     // for MakeCachePrefix

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // for AssertGPUSupport
#endif

namespace xgboost::data {
ExtMemQuantileDMatrix::ExtMemQuantileDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy,
                                             std::shared_ptr<DMatrix> ref,
                                             DataIterResetCallback *reset,
                                             XGDMatrixCallbackNext *next, bst_bin_t max_bin,
                                             std::int64_t max_quantile_blocks,
                                             ExtMemConfig const &config)
    : cache_prefix_{config.cache}, on_host_{config.on_host} {
  cache_prefix_ = MakeCachePrefix(cache_prefix_);
  auto iter = std::make_shared<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>>(
      iter_handle, reset, next);
  iter->Reset();
  // Fetch the first iter
  bool valid = iter->Next();
  CHECK(valid) << "Qauntile DMatrix must have at least 1 batch.";

  auto pctx = MakeProxy(proxy)->Ctx();
  Context ctx;
  ctx.Init(Args{{"nthread", std::to_string(config.n_threads)}, {"device", pctx->DeviceName()}});

  BatchParam p{max_bin, tree::TrainParam::DftSparseThreshold()};
  if (ctx.IsCPU()) {
    this->InitFromCPU(&ctx, iter, proxy, p, config.missing, ref);
  } else {
    p.n_prefetch_batches = ::xgboost::cuda_impl::DftPrefetchBatches();
    this->InitFromCUDA(&ctx, iter, proxy, p, ref, max_quantile_blocks, config);
  }
  this->batch_ = p;
  this->fmat_ctx_ = ctx;
}

ExtMemQuantileDMatrix::~ExtMemQuantileDMatrix() {
  // Clear out all resources before deleting the cache file.
  ghist_index_source_.reset();
  std::visit([](auto &&ptr) { ptr.reset(); }, ellpack_page_source_);

  DeleteCacheFiles(cache_info_);
}

BatchSet<ExtSparsePage> ExtMemQuantileDMatrix::GetExtBatches(Context const *, BatchParam const &) {
  LOG(FATAL) << "Not implemented for `ExtMemQuantileDMatrix`.";
  auto begin_iter =
      BatchIterator<ExtSparsePage>(new SimpleBatchIteratorImpl<ExtSparsePage>(nullptr));
  return BatchSet<ExtSparsePage>{begin_iter};
}

void ExtMemQuantileDMatrix::InitFromCPU(
    Context const *ctx,
    std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> iter,
    DMatrixHandle proxy_handle, BatchParam const &p, float missing, std::shared_ptr<DMatrix> ref) {
  xgboost_NVTX_FN_RANGE();

  auto proxy = MakeProxy(proxy_handle);
  CHECK(proxy);

  common::HistogramCuts cuts;
  ExternalDataInfo ext_info;
  cpu_impl::GetDataShape(ctx, proxy, *iter, missing, &ext_info);
  ext_info.SetInfo(ctx, &this->info_);

  this->n_batches_ = ext_info.n_batches;

  /**
   * Generate quantiles
   */
  std::vector<FeatureType> h_ft;
  cpu_impl::MakeSketches(ctx, iter.get(), proxy, ref, missing, &cuts, p, this->info_, ext_info,
                         &h_ft);

  /**
   * Generate gradient index
   */
  auto id = MakeCache(this, ".gradient_index.page", false, cache_prefix_, &cache_info_);
  this->ghist_index_source_ = std::make_unique<ExtGradientIndexPageSource>(
      ctx, missing, &this->info_, cache_info_.at(id), p, cuts, iter, proxy, ext_info.base_rowids);

  /**
   * Force initialize the cache and do some sanity checks along the way
   */
  bst_idx_t batch_cnt = 0, k = 0;
  bst_idx_t n_total_samples = 0;
  for (auto const &page : this->GetGradientIndexImpl()) {
    n_total_samples += page.Size();
    CHECK_EQ(page.base_rowid, ext_info.base_rowids[k]);
    CHECK_EQ(page.Features(), this->info_.num_col_);
    ++k, ++batch_cnt;
  }
  CHECK_EQ(batch_cnt, ext_info.n_batches);
  CHECK_EQ(n_total_samples, ext_info.accumulated_rows);
  if (cuts.HasCategorical()) {
    CHECK(!this->info_.feature_types.Empty());
  }
  CHECK_EQ(cuts.HasCategorical(), this->info_.HasCategorical());
}

[[nodiscard]] BatchSet<GHistIndexMatrix> ExtMemQuantileDMatrix::GetGradientIndexImpl() {
  return BatchSet{BatchIterator<GHistIndexMatrix>{this->ghist_index_source_}};
}

BatchSet<GHistIndexMatrix> ExtMemQuantileDMatrix::GetGradientIndex(Context const *,
                                                                   BatchParam const &param) {
  if (param.Initialized()) {
    detail::CheckParam(this->batch_, param);
    CHECK(!detail::RegenGHist(param, batch_)) << error::InconsistentMaxBin();
  }

  CHECK(this->ghist_index_source_)
      << "The `ExtMemQuantileDMatrix` is initialized using GPU data, cannot be used for CPU.";
  this->ghist_index_source_->Reset(param);

  if (!std::isnan(param.sparse_thresh) &&
      param.sparse_thresh != tree::TrainParam::DftSparseThreshold()) {
    LOG(WARNING) << "`sparse_threshold` can not be changed when `QuantileDMatrix` is used instead "
                    "of `DMatrix`.";
  }

  return this->GetGradientIndexImpl();
}

#if !defined(XGBOOST_USE_CUDA)
void ExtMemQuantileDMatrix::InitFromCUDA(
    Context const *, std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>>,
    DMatrixHandle, BatchParam const &, std::shared_ptr<DMatrix>, std::int64_t,
    ExtMemConfig const &) {
  common::AssertGPUSupport();
}

BatchSet<EllpackPage> ExtMemQuantileDMatrix::GetEllpackBatches(Context const *,
                                                               const BatchParam &) {
  common::AssertGPUSupport();
  auto batch_set =
      std::visit([this](auto &&ptr) { return BatchSet{BatchIterator<EllpackPage>{ptr}}; },
                 this->ellpack_page_source_);
  return batch_set;
}

BatchSet<EllpackPage> ExtMemQuantileDMatrix::GetEllpackPageImpl() {
  common::AssertGPUSupport();
  auto batch_set =
      std::visit([this](auto &&ptr) { return BatchSet{BatchIterator<EllpackPage>{ptr}}; },
                 this->ellpack_page_source_);
  return batch_set;
}
#endif
}  // namespace xgboost::data
