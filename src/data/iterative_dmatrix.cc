/**
 * Copyright 2022-2024, XGBoost contributors
 */
#include "iterative_dmatrix.h"

#include <algorithm>  // for copy
#include <cstddef>    // for size_t
#include <memory>     // for shared_ptr
#include <utility>    // for move
#include <vector>     // for vector

#include "../common/categorical.h"  // common::IsCat
#include "../common/hist_util.h"    // for HistogramCuts
#include "../tree/param.h"          // FIXME(jiamingy): Find a better way to share this parameter.
#include "batch_utils.h"            // for RegenGHist
#include "gradient_index.h"         // for GHistIndexMatrix
#include "proxy_dmatrix.h"          // for DataIterProxy
#include "quantile_dmatrix.h"       // for GetCutsFromRef
#include "quantile_dmatrix.h"       // for GetDataShape, MakeSketches
#include "simple_batch_iterator.h"  // for SimpleBatchIteratorImpl
#include "xgboost/data.h"           // for FeatureType, DMatrix
#include "xgboost/logging.h"

namespace xgboost::data {
IterativeDMatrix::IterativeDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy,
                                   std::shared_ptr<DMatrix> ref, DataIterResetCallback* reset,
                                   XGDMatrixCallbackNext* next, float missing, int nthread,
                                   bst_bin_t max_bin, std::int64_t max_quantile_blocks)
    : proxy_{proxy}, reset_{reset}, next_{next} {
  // fetch the first batch
  auto iter =
      DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{iter_handle, reset_, next_};
  iter.Reset();
  bool valid = iter.Next();
  CHECK(valid) << "Iterative DMatrix must have at least 1 batch.";

  auto pctx = MakeProxy(proxy_)->Ctx();

  Context ctx;
  ctx.Init(Args{{"nthread", std::to_string(nthread)}, {"device", pctx->DeviceName()}});
  // hardcoded parameter.
  BatchParam p{max_bin, tree::TrainParam::DftSparseThreshold()};

  if (ctx.IsCUDA()) {
    this->InitFromCUDA(&ctx, p, max_quantile_blocks, iter_handle, missing, ref);
  } else {
    this->InitFromCPU(&ctx, p, iter_handle, missing, ref);
  }

  this->fmat_ctx_ = ctx;
  this->batch_ = p;

  LOG(INFO) << "Finished constructing the `IterativeDMatrix`: (" << this->Info().num_row_ << ", "
            << this->Info().num_col_ << ", " << this->Info().num_nonzero_ << ").";
}

void IterativeDMatrix::InitFromCPU(Context const* ctx, BatchParam const& p,
                                   DataIterHandle iter_handle, float missing,
                                   std::shared_ptr<DMatrix> ref) {
  DMatrixProxy* proxy = MakeProxy(proxy_);
  CHECK(proxy);

  // The external iterator
  auto iter =
      DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{iter_handle, reset_, next_};
  common::HistogramCuts cuts;
  ExternalDataInfo ext_info;
  cpu_impl::GetDataShape(ctx, proxy, iter, missing, &ext_info);
  ext_info.SetInfo(ctx, &this->info_);

  /**
   * Generate quantiles
   */
  std::vector<FeatureType> h_ft;
  cpu_impl::MakeSketches(ctx, &iter, proxy, ref, missing, &cuts, p, this->info_, ext_info, &h_ft);

  /**
   * Generate gradient index.
   */
  this->ghist_ = std::make_unique<GHistIndexMatrix>(this->info_, std::move(cuts), p.max_bin);
  std::size_t rbegin = 0;
  std::size_t prev_sum = 0;
  std::size_t i = 0;
  while (iter.Next()) {
    HostAdapterDispatch(proxy, [&](auto const& batch) {
      proxy->Info().num_nonzero_ = ext_info.batch_nnz[i];
      this->ghist_->PushAdapterBatch(ctx, rbegin, prev_sum, batch, missing, h_ft, p.sparse_thresh,
                                     Info().num_row_);
    });
    if (ext_info.n_batches != 1) {
      this->info_.Extend(std::move(proxy->Info()), false, true);
    }
    auto batch_size = BatchSamples(proxy);
    prev_sum = this->ghist_->row_ptr[rbegin + batch_size];
    rbegin += batch_size;
    ++i;
  }
  iter.Reset();
  CHECK_EQ(rbegin, Info().num_row_);
  CHECK_EQ(this->ghist_->Features(), Info().num_col_);

  /**
   * Generate column matrix
   */
  bst_idx_t accumulated_rows = 0;
  while (iter.Next()) {
    HostAdapterDispatch(proxy, [&](auto const& batch) {
      this->ghist_->PushAdapterBatchColumns(ctx, batch, missing, accumulated_rows);
    });
    accumulated_rows += BatchSamples(proxy);
  }
  iter.Reset();
  CHECK_EQ(accumulated_rows, Info().num_row_);

  if (ext_info.n_batches == 1) {
    this->info_ = std::move(proxy->Info());
    this->info_.num_nonzero_ = ext_info.nnz;
    this->info_.num_col_ = ext_info.n_features;  // proxy might be empty.
    CHECK_EQ(proxy->Info().labels.Size(), 0);
  }

  info_.feature_types.HostVector() = h_ft;
}

BatchSet<GHistIndexMatrix> IterativeDMatrix::GetGradientIndex(Context const* ctx,
                                                              BatchParam const& param) {
  if (param.Initialized()) {
    detail::CheckParam(this->batch_, param);
    CHECK(!detail::RegenGHist(param, batch_)) << error::InconsistentMaxBin();
  }
  if (!ellpack_ && !ghist_) {
    LOG(FATAL) << "`QuantileDMatrix` not initialized.";
  }

  if (!ghist_) {
    if (!ctx->IsCUDA()) {
      ghist_ = std::make_shared<GHistIndexMatrix>(ctx, Info(), *ellpack_, param);
    } else if (!fmat_ctx_.IsCUDA()) {
      ghist_ = std::make_shared<GHistIndexMatrix>(&fmat_ctx_, Info(), *ellpack_, param);
    } else {
      // Can happen when QDM is initialized on GPU, but a CPU version is queried by a different QDM
      // for cut reference.
      auto cpu_ctx = ctx->MakeCPU();
      ghist_ = std::make_shared<GHistIndexMatrix>(&cpu_ctx, Info(), *ellpack_, param);
    }
  }

  if (!std::isnan(param.sparse_thresh) &&
      param.sparse_thresh != tree::TrainParam::DftSparseThreshold()) {
    LOG(WARNING) << "`sparse_threshold` can not be changed when `QuantileDMatrix` is used instead "
                    "of `DMatrix`.";
  }

  auto begin_iter =
      BatchIterator<GHistIndexMatrix>(new SimpleBatchIteratorImpl<GHistIndexMatrix>(ghist_));
  return BatchSet<GHistIndexMatrix>(begin_iter);
}

BatchSet<ExtSparsePage> IterativeDMatrix::GetExtBatches(Context const* ctx,
                                                        BatchParam const& param) {
  for (auto const& page : this->GetGradientIndex(ctx, param)) {
    auto p_out = std::make_shared<SparsePage>();
    p_out->data.Resize(this->Info().num_nonzero_);
    p_out->offset.Resize(this->Info().num_row_ + 1);

    auto& h_offset = p_out->offset.HostVector();
    CHECK_EQ(page.row_ptr.size(), h_offset.size());
    std::copy(page.row_ptr.cbegin(), page.row_ptr.cend(), h_offset.begin());

    auto& h_data = p_out->data.HostVector();
    auto const& vals = page.cut.Values();
    auto const& mins = page.cut.MinValues();
    auto const& ptrs = page.cut.Ptrs();
    auto ft = Info().feature_types.ConstHostSpan();

    AssignColumnBinIndex(page, [&](auto bin_idx, std::size_t idx, std::size_t, bst_feature_t fidx) {
      float v;
      if (common::IsCat(ft, fidx)) {
        v = vals[bin_idx];
      } else {
        v = common::HistogramCuts::NumericBinValue(ptrs, vals, mins, fidx, bin_idx);
      }
      h_data[idx] = Entry{fidx, v};
    });

    auto p_ext_out = std::make_shared<ExtSparsePage>(p_out);
    auto begin_iter =
        BatchIterator<ExtSparsePage>(new SimpleBatchIteratorImpl<ExtSparsePage>(p_ext_out));
    return BatchSet<ExtSparsePage>(begin_iter);
  }
  LOG(FATAL) << "Unreachable";
  auto begin_iter =
      BatchIterator<ExtSparsePage>(new SimpleBatchIteratorImpl<ExtSparsePage>(nullptr));
  return BatchSet<ExtSparsePage>(begin_iter);
}

#if !defined(XGBOOST_USE_CUDA)
inline void IterativeDMatrix::InitFromCUDA(Context const*, BatchParam const&, std::int64_t,
                                           DataIterHandle, float, std::shared_ptr<DMatrix>) {
  // silent the warning about unused variables.
  (void)(proxy_);
  (void)(reset_);
  (void)(next_);
  common::AssertGPUSupport();
}

inline BatchSet<EllpackPage> IterativeDMatrix::GetEllpackBatches(Context const*,
                                                                 BatchParam const&) {
  common::AssertGPUSupport();
  auto begin_iter = BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(ellpack_));
  return BatchSet<EllpackPage>(BatchIterator<EllpackPage>(begin_iter));
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data
