/**
 * Copyright 2014-2023 by XGBoost Contributors
 * \file sparse_page_dmatrix.cc
 *
 * \brief The external memory version of Page Iterator.
 * \author Tianqi Chen
 */
#include "./sparse_page_dmatrix.h"

#include "../collective/communicator-inl.h"
#include "./simple_batch_iterator.h"
#include "batch_utils.h"  // for RegenGHist
#include "gradient_index.h"

namespace xgboost::data {
MetaInfo &SparsePageDMatrix::Info() { return info_; }

const MetaInfo &SparsePageDMatrix::Info() const { return info_; }

namespace detail {
// Use device dispatch
std::size_t NSamplesDevice(DMatrixProxy *)  // NOLINT
#if defined(XGBOOST_USE_CUDA)
;  // NOLINT
#else
{
  common::AssertGPUSupport();
  return 0;
}
#endif
std::size_t NFeaturesDevice(DMatrixProxy *)  // NOLINT
#if defined(XGBOOST_USE_CUDA)
;  // NOLINT
#else
{
  common::AssertGPUSupport();
  return 0;
}
#endif
}  // namespace detail


SparsePageDMatrix::SparsePageDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy_handle,
                                     DataIterResetCallback *reset,
                                     XGDMatrixCallbackNext *next, float missing,
                                     int32_t nthreads, std::string cache_prefix)
    : proxy_{proxy_handle}, iter_{iter_handle}, reset_{reset}, next_{next}, missing_{missing},
      cache_prefix_{std::move(cache_prefix)} {
  Context ctx;
  ctx.nthread = nthreads;

  cache_prefix_ = cache_prefix_.empty() ? "DMatrix" : cache_prefix_;
  if (collective::IsDistributed()) {
    cache_prefix_ += ("-r" + std::to_string(collective::GetRank()));
  }
  DMatrixProxy *proxy = MakeProxy(proxy_);
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};

  uint32_t n_batches = 0;
  size_t n_features = 0;
  size_t n_samples = 0;
  size_t nnz = 0;

  auto num_rows = [&]() {
    bool type_error {false};
    size_t n_samples = HostAdapterDispatch(
        proxy, [](auto const &value) { return value.NumRows(); }, &type_error);
    if (type_error) {
      n_samples = detail::NSamplesDevice(proxy);
    }
    return n_samples;
  };
  auto num_cols = [&]() {
    bool type_error {false};
    size_t n_features = HostAdapterDispatch(
        proxy, [](auto const &value) { return value.NumCols(); }, &type_error);
    if (type_error) {
      n_features = detail::NFeaturesDevice(proxy);
    }
    return n_features;
  };

  // the proxy is iterated together with the sparse page source so we can obtain all
  // information in 1 pass.
  for (auto const &page : this->GetRowBatchesImpl(&ctx)) {
    this->info_.Extend(std::move(proxy->Info()), false, false);
    n_features = std::max(n_features, num_cols());
    n_samples += num_rows();
    nnz += page.data.Size();
    n_batches++;
  }

  iter.Reset();

  this->n_batches_ = n_batches;
  this->info_.num_row_ = n_samples;
  this->info_.num_col_ = n_features;
  this->info_.num_nonzero_ = nnz;

  info_.SynchronizeNumberOfColumns();
  CHECK_NE(info_.num_col_, 0);

  fmat_ctx_ = ctx;
}

void SparsePageDMatrix::InitializeSparsePage(Context const *ctx) {
  auto id = MakeCache(this, ".row.page", cache_prefix_, &cache_info_);
  // Don't use proxy DMatrix once this is already initialized, this allows users to
  // release the iterator and data.
  if (cache_info_.at(id)->written) {
    CHECK(sparse_page_source_);
    sparse_page_source_->Reset();
    return;
  }

  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{iter_, reset_, next_};
  DMatrixProxy *proxy = MakeProxy(proxy_);
  sparse_page_source_.reset();  // clear before creating new one to prevent conflicts.
  sparse_page_source_ = std::make_shared<SparsePageSource>(iter, proxy, this->missing_,
                                                           ctx->Threads(), this->info_.num_col_,
                                                           this->n_batches_, cache_info_.at(id));
}

BatchSet<SparsePage> SparsePageDMatrix::GetRowBatchesImpl(Context const* ctx) {
  this->InitializeSparsePage(ctx);
  auto begin_iter = BatchIterator<SparsePage>(sparse_page_source_);
  return BatchSet<SparsePage>(BatchIterator<SparsePage>(begin_iter));
}

BatchSet<SparsePage> SparsePageDMatrix::GetRowBatches() {
  // Use context from initialization for the default row page.
  return this->GetRowBatchesImpl(&fmat_ctx_);
}

BatchSet<CSCPage> SparsePageDMatrix::GetColumnBatches(Context const *ctx) {
  auto id = MakeCache(this, ".col.page", cache_prefix_, &cache_info_);
  CHECK_NE(this->Info().num_col_, 0);
  this->InitializeSparsePage(ctx);
  if (!column_source_) {
    column_source_ =
        std::make_shared<CSCPageSource>(this->missing_, ctx->Threads(), this->Info().num_col_,
                                        this->n_batches_, cache_info_.at(id), sparse_page_source_);
  } else {
    column_source_->Reset();
  }
  auto begin_iter = BatchIterator<CSCPage>(column_source_);
  return BatchSet<CSCPage>(BatchIterator<CSCPage>(begin_iter));
}

BatchSet<SortedCSCPage> SparsePageDMatrix::GetSortedColumnBatches(Context const *ctx) {
  auto id = MakeCache(this, ".sorted.col.page", cache_prefix_, &cache_info_);
  CHECK_NE(this->Info().num_col_, 0);
  this->InitializeSparsePage(ctx);
  if (!sorted_column_source_) {
    sorted_column_source_ = std::make_shared<SortedCSCPageSource>(
        this->missing_, ctx->Threads(), this->Info().num_col_, this->n_batches_, cache_info_.at(id),
        sparse_page_source_);
  } else {
    sorted_column_source_->Reset();
  }
  auto begin_iter = BatchIterator<SortedCSCPage>(sorted_column_source_);
  return BatchSet<SortedCSCPage>(BatchIterator<SortedCSCPage>(begin_iter));
}

BatchSet<GHistIndexMatrix> SparsePageDMatrix::GetGradientIndex(Context const *ctx,
                                                               const BatchParam &param) {
  CHECK_GE(param.max_bin, 2);
  auto id = MakeCache(this, ".gradient_index.page", cache_prefix_, &cache_info_);
  this->InitializeSparsePage(ctx);
  if (!cache_info_.at(id)->written || detail::RegenGHist(batch_param_, param)) {
    cache_info_.erase(id);
    MakeCache(this, ".gradient_index.page", cache_prefix_, &cache_info_);
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
        this->missing_, ctx->Threads(), this->Info().num_col_, this->n_batches_, cache_info_.at(id),
        param, std::move(cuts), this->IsDense(), ft, sparse_page_source_));
  } else {
    CHECK(ghist_index_source_);
    ghist_index_source_->Reset();
  }
  auto begin_iter = BatchIterator<GHistIndexMatrix>(ghist_index_source_);
  return BatchSet<GHistIndexMatrix>(BatchIterator<GHistIndexMatrix>(begin_iter));
}

#if !defined(XGBOOST_USE_CUDA)
BatchSet<EllpackPage> SparsePageDMatrix::GetEllpackBatches(Context const *, const BatchParam &) {
  common::AssertGPUSupport();
  auto begin_iter = BatchIterator<EllpackPage>(ellpack_page_source_);
  return BatchSet<EllpackPage>(BatchIterator<EllpackPage>(begin_iter));
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data
