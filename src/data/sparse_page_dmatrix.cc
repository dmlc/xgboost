/**
 * Copyright 2014-2024, XGBoost Contributors
 * \file sparse_page_dmatrix.cc
 *
 * \brief The external memory version of Page Iterator.
 * \author Tianqi Chen
 */
#include "sparse_page_dmatrix.h"

#include <algorithm>  // for max
#include <memory>     // for make_shared
#include <string>     // for string
#include <utility>    // for move
#include <variant>    // for visit

#include "batch_utils.h"         // for RegenGHist
#include "gradient_index.h"      // for GHistIndexMatrix
#include "sparse_page_source.h"  // for MakeCachePrefix

namespace xgboost::data {
MetaInfo &SparsePageDMatrix::Info() { return info_; }

const MetaInfo &SparsePageDMatrix::Info() const { return info_; }

SparsePageDMatrix::SparsePageDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy_handle,
                                     DataIterResetCallback *reset, XGDMatrixCallbackNext *next,
                                     ExtMemConfig const &config)
    : proxy_{proxy_handle},
      iter_{iter_handle},
      reset_{reset},
      next_{next},
      missing_{config.missing},
      cache_prefix_{config.cache},
      on_host_{config.on_host},
      min_cache_page_bytes_{config.min_cache_page_bytes} {
  Context ctx;
  ctx.Init(Args{{"nthread", std::to_string(config.n_threads)}});
  cache_prefix_ = MakeCachePrefix(cache_prefix_);

  DMatrixProxy *proxy = MakeProxy(proxy_);
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};

  // The proxy is iterated together with the sparse page source so we can obtain all
  // information in 1 pass.
  for (auto const &page : this->GetRowBatchesImpl(&ctx)) {
    this->info_.Extend(std::move(proxy->Info()), false, false);
    ext_info_.n_features =
        std::max(static_cast<bst_feature_t>(ext_info_.n_features), BatchColumns(proxy));
    ext_info_.accumulated_rows += BatchSamples(proxy);
    ext_info_.nnz += page.data.Size();
    ext_info_.n_batches++;
    ext_info_.base_rowids.push_back(page.Size());
    ext_info_.batch_nnz.push_back(page.data.Size());
  }
  std::partial_sum(ext_info_.base_rowids.cbegin(), ext_info_.base_rowids.cend(),
                   ext_info_.base_rowids.begin());

  iter.Reset();

  ext_info_.SetInfo(&ctx, &this->info_);

  fmat_ctx_ = ctx;
}

SparsePageDMatrix::~SparsePageDMatrix() {
  // Clear out all resources before deleting the cache file.
  sparse_page_source_.reset();
  std::visit([](auto &&ptr) { ptr.reset(); }, ellpack_page_source_);
  column_source_.reset();
  sorted_column_source_.reset();
  ghist_index_source_.reset();

  DeleteCacheFiles(cache_info_);
}

void SparsePageDMatrix::InitializeSparsePage(Context const *ctx) {
  auto id = MakeCache(this, ".row.page", false, cache_prefix_, &cache_info_);
  // Don't use proxy DMatrix once this is already initialized, this allows users to
  // release the iterator and data.
  if (cache_info_.at(id)->written) {
    CHECK(this->sparse_page_source_);
    this->sparse_page_source_->Reset({});
    return;
  }

  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{iter_, reset_, next_};
  DMatrixProxy *proxy = MakeProxy(proxy_);
  sparse_page_source_.reset();  // clear before creating new one to prevent conflicts.
  // During initialization, the n_batches is 0.
  CHECK_EQ(this->ext_info_.n_batches, static_cast<decltype(this->ext_info_.n_batches)>(0));
  sparse_page_source_ = std::make_shared<SparsePageSource>(
      iter, proxy, this->missing_, ctx->Threads(), this->info_.num_col_, this->ext_info_.n_batches,
      cache_info_.at(id));
}

BatchSet<SparsePage> SparsePageDMatrix::GetRowBatchesImpl(Context const *ctx) {
  this->InitializeSparsePage(ctx);
  return BatchSet{BatchIterator<SparsePage>{this->sparse_page_source_}};
}

BatchSet<SparsePage> SparsePageDMatrix::GetRowBatches() {
  // Use context from initialization for the default row page.
  return this->GetRowBatchesImpl(&fmat_ctx_);
}

BatchSet<CSCPage> SparsePageDMatrix::GetColumnBatches(Context const *ctx) {
  auto id = MakeCache(this, ".col.page", false, cache_prefix_, &cache_info_);
  CHECK_NE(this->Info().num_col_, 0);
  this->InitializeSparsePage(ctx);
  if (!column_source_) {
    column_source_ = std::make_shared<CSCPageSource>(this->missing_, ctx->Threads(),
                                                     this->Info().num_col_, this->NumBatches(),
                                                     cache_info_.at(id), sparse_page_source_);
  } else {
    column_source_->Reset({});
  }
  return BatchSet{BatchIterator<CSCPage>{this->column_source_}};
}

BatchSet<SortedCSCPage> SparsePageDMatrix::GetSortedColumnBatches(Context const *ctx) {
  auto id = MakeCache(this, ".sorted.col.page", false, cache_prefix_, &cache_info_);
  CHECK_NE(this->Info().num_col_, 0);
  this->InitializeSparsePage(ctx);
  if (!sorted_column_source_) {
    sorted_column_source_ = std::make_shared<SortedCSCPageSource>(
        this->missing_, ctx->Threads(), this->Info().num_col_, this->NumBatches(),
        cache_info_.at(id), sparse_page_source_);
  } else {
    sorted_column_source_->Reset({});
  }
  return BatchSet{BatchIterator<SortedCSCPage>{this->sorted_column_source_}};
}

BatchSet<GHistIndexMatrix> SparsePageDMatrix::GetGradientIndex(Context const *ctx,
                                                               const BatchParam &param) {
  if (param.Initialized()) {
    CHECK_GE(param.max_bin, 2);
  }
  detail::CheckEmpty(batch_param_, param);
  auto id = MakeCache(this, ".gradient_index.page", false, cache_prefix_, &cache_info_);
  if (!cache_info_.at(id)->written || detail::RegenGHist(batch_param_, param)) {
    this->InitializeSparsePage(ctx);
    cache_info_.erase(id);
    id = MakeCache(this, ".gradient_index.page", false, cache_prefix_, &cache_info_);
    LOG(INFO) << "Generating new Gradient Index.";
    // Use sorted sketch for approx.
    auto sorted_sketch = param.regen;
    auto cuts = common::SketchOnDMatrix(ctx, this, param.max_bin, sorted_sketch, param.hess);
    this->InitializeSparsePage(ctx);  // reset after use.

    batch_param_ = param;
    ghist_index_source_.reset();
    CHECK_NE(cuts.Values().size(), 0);
    auto ft = this->info_.feature_types.ConstHostSpan();
    ghist_index_source_.reset(new GradientIndexPageSource(
        this->missing_, ctx->Threads(), this->Info().num_col_, this->NumBatches(),
        cache_info_.at(id), param, std::move(cuts), this->IsDense(), ft, sparse_page_source_));
  } else {
    CHECK(ghist_index_source_);
    ghist_index_source_->Reset(param);
  }
  return BatchSet{BatchIterator<GHistIndexMatrix>{this->ghist_index_source_}};
}

#if !defined(XGBOOST_USE_CUDA)
BatchSet<EllpackPage> SparsePageDMatrix::GetEllpackBatches(Context const *, const BatchParam &) {
  common::AssertGPUSupport();
  return BatchSet{BatchIterator<EllpackPage>{nullptr}};
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data
