/*!
 * Copyright 2014-2021 by Contributors
 * \file sparse_page_dmatrix.cc
 * \brief The external memory version of Page Iterator.
 * \author Tianqi Chen
 */
#include "./sparse_page_dmatrix.h"
#include "./simple_batch_iterator.h"
#include "gradient_index.h"

namespace xgboost {
namespace data {

MetaInfo &SparsePageDMatrix::Info() { return info_; }

const MetaInfo &SparsePageDMatrix::Info() const { return info_; }

SparsePageDMatrix::SparsePageDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy_handle,
                                     DataIterResetCallback *reset,
                                     XGDMatrixCallbackNext *next, float missing,
                                     int32_t nthreads, std::string cache_prefix)
    : proxy_{proxy_handle}, iter_{iter_handle}, reset_{reset}, next_{next}, missing_{missing},
      nthreads_{nthreads}, cache_prefix_{std::move(cache_prefix)} {
  DMatrixProxy *proxy = MakeProxy(proxy_);
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};

  size_t n_batches = 0;
  size_t n_features = 0;
  size_t n_samples = 0;
  size_t nnz = 0;

  auto num_rows = [&]() {
    return HostAdapterDispatch(
        proxy, [](auto const &value) { return value.NumRows(); });
  };
  auto num_cols = [&]() {
    return HostAdapterDispatch(
        proxy, [](auto const &value) { return value.NumCols(); });
  };
  // the proxy is iterated together with the sparse page source so we can obtain all
  // information in 1 pass.
  for (auto const &page : this->GetRowBatchesImpl()) {
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

  rabit::Allreduce<rabit::op::Max>(&info_.num_col_, 1);
  CHECK_NE(info_.num_col_, 0);
}

void SparsePageDMatrix::InitializeSparsePage() {
  auto id = MakeCache(this, ".row.page", cache_prefix_, &cache_info_);
  // Don't use proxy DMatrix once this is already initialized, this allows users to
  // release the iterator and data.
  if (cache_info_.at(id)->written) {
    CHECK(sparse_page_source_);
    sparse_page_source_->Reset();
    return;
  }

  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};
  DMatrixProxy *proxy = MakeProxy(proxy_);
  sparse_page_source_.reset();  // clear before creating new one to prevent conflicts.
  sparse_page_source_ = std::make_shared<SparsePageSource>(
      iter, proxy, this->missing_, this->nthreads_, this->info_.num_col_,
      this->n_batches_, cache_info_.at(id));
}

BatchSet<SparsePage> SparsePageDMatrix::GetRowBatchesImpl() {
  this->InitializeSparsePage();
  auto begin_iter = BatchIterator<SparsePage>(sparse_page_source_);
  return BatchSet<SparsePage>(BatchIterator<SparsePage>(begin_iter));
}

BatchSet<SparsePage> SparsePageDMatrix::GetRowBatches() {
  return this->GetRowBatchesImpl();
}

BatchSet<CSCPage> SparsePageDMatrix::GetColumnBatches() {
  auto id = MakeCache(this, ".col.page", cache_prefix_, &cache_info_);
  CHECK_NE(this->Info().num_col_, 0);
  this->InitializeSparsePage();
  if (!column_source_) {
    column_source_ = std::make_shared<CSCPageSource>(
        this->missing_, this->nthreads_, this->Info().num_col_,
        this->n_batches_, cache_info_.at(id), sparse_page_source_);
  } else {
    column_source_->Reset();
  }
  auto begin_iter = BatchIterator<CSCPage>(column_source_);
  return BatchSet<CSCPage>(BatchIterator<CSCPage>(begin_iter));
}

BatchSet<SortedCSCPage> SparsePageDMatrix::GetSortedColumnBatches() {
  auto id = MakeCache(this, ".sorted.col.page", cache_prefix_, &cache_info_);
  CHECK_NE(this->Info().num_col_, 0);
  this->InitializeSparsePage();
  if (!sorted_column_source_) {
    sorted_column_source_ = std::make_shared<SortedCSCPageSource>(
        this->missing_, this->nthreads_, this->Info().num_col_,
        this->n_batches_, cache_info_.at(id), sparse_page_source_);
  } else {
    sorted_column_source_->Reset();
  }
  auto begin_iter = BatchIterator<SortedCSCPage>(sorted_column_source_);
  return BatchSet<SortedCSCPage>(BatchIterator<SortedCSCPage>(begin_iter));
}

BatchSet<GHistIndexMatrix> SparsePageDMatrix::GetGradientIndex(const BatchParam& param) {
  CHECK_GE(param.max_bin, 2);
  LOG(WARNING) << "External memory support for hist is working in progress.";
  // External memory is not support
  if (!ghist_index_source_ || (param != batch_param_ && param != BatchParam{})) {
    ghist_index_source_.reset(new GHistIndexMatrix{this, param.max_bin});
    batch_param_ = param;
  }
  auto begin_iter = BatchIterator<GHistIndexMatrix>(
      new SimpleBatchIteratorImpl<GHistIndexMatrix>(ghist_index_source_.get()));
  return BatchSet<GHistIndexMatrix>(begin_iter);
}

#if !defined(XGBOOST_USE_CUDA)
BatchSet<EllpackPage> SparsePageDMatrix::GetEllpackBatches(const BatchParam& param) {
  common::AssertGPUSupport();
  auto begin_iter = BatchIterator<EllpackPage>(ellpack_page_source_);
  return BatchSet<EllpackPage>(BatchIterator<EllpackPage>(begin_iter));
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace data
}  // namespace xgboost
