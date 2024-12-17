/**
 * Copyright 2020-2024, XGBoost contributors
 */
#include <memory>     // for shared_ptr
#include <utility>    // for move

#include "batch_utils.h"  // for RegenGHist, CheckParam
#include "device_adapter.cuh"
#include "ellpack_page.cuh"
#include "iterative_dmatrix.h"
#include "proxy_dmatrix.cuh"
#include "proxy_dmatrix.h"  // for BatchSamples, BatchColumns
#include "simple_batch_iterator.h"

namespace xgboost::data {
void IterativeDMatrix::InitFromCUDA(Context const* ctx, BatchParam const& p,
                                    std::int64_t max_quantile_blocks, DataIterHandle iter_handle,
                                    float missing, std::shared_ptr<DMatrix> ref) {
  // A handle passed to external iterator.
  DMatrixProxy* proxy = MakeProxy(proxy_);
  CHECK(proxy);

  // The external iterator
  auto iter =
      DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{iter_handle, reset_, next_};

  dh::XGBCachingDeviceAllocator<char> alloc;

  // Sketch for all batches.

  std::int32_t current_device{dh::CurrentDevice()};
  auto get_ctx = [&]() {
    Context d_ctx = (ctx->IsCUDA()) ? *ctx : Context{}.MakeCUDA(current_device);
    CHECK(!d_ctx.IsCPU());
    return d_ctx;
  };

  fmat_ctx_ = get_ctx();

  /**
   * Generate quantiles
   */
  auto cuts = std::make_shared<common::HistogramCuts>();
  ExternalDataInfo ext_info;
  cuda_impl::MakeSketches(ctx, &iter, proxy, ref, p, missing, cuts, this->info_,
                          max_quantile_blocks, &ext_info);
  ext_info.SetInfo(ctx, &this->info_);

  auto init_page = [this, &cuts, &ext_info]() {
    if (!ellpack_) {
      // Should be put inside the while loop to protect against empty batch.  In
      // that case device id is invalid.
      ellpack_.reset(new EllpackPage);
      *(ellpack_->Impl()) = EllpackPageImpl(&fmat_ctx_, cuts, this->IsDense(), ext_info.row_stride,
                                            ext_info.accumulated_rows);
    }
  };

  /**
   * Generate gradient index.
   */
  bst_idx_t offset = 0;
  iter.Reset();
  bst_idx_t n_batches_for_verification = 0;
  while (iter.Next()) {
    init_page();
    dh::safe_cuda(cudaSetDevice(dh::GetDevice(ctx).ordinal));
    auto rows = BatchSamples(proxy);
    dh::device_vector<size_t> row_counts(rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(), row_counts.size());
    cuda_impl::Dispatch(proxy, [=](auto const& value) {
      return GetRowCounts(ctx, value, row_counts_span, dh::GetDevice(ctx), missing);
    });
    auto is_dense = this->IsDense();

    proxy->Info().feature_types.SetDevice(dh::GetDevice(ctx));
    auto d_feature_types = proxy->Info().feature_types.ConstDeviceSpan();
    auto new_impl = cuda_impl::Dispatch(proxy, [&](auto const& value) {
      return EllpackPageImpl{
          &fmat_ctx_,          value, missing, is_dense, row_counts_span, d_feature_types,
          ext_info.row_stride, rows,  cuts};
    });
    bst_idx_t num_elements = ellpack_->Impl()->Copy(&fmat_ctx_, &new_impl, offset);
    offset += num_elements;

    proxy->Info().num_row_ = BatchSamples(proxy);
    proxy->Info().num_col_ = ext_info.n_features;
    if (ext_info.n_batches != 1) {
      this->info_.Extend(std::move(proxy->Info()), false, true);
    }
    n_batches_for_verification++;
  }
  CHECK_EQ(ext_info.n_batches, n_batches_for_verification)
      << "Different number of batches returned between 2 iterations";

  if (ext_info.n_batches == 1) {
    this->info_ = std::move(proxy->Info());
    this->info_.num_nonzero_ = ext_info.nnz;
    CHECK_EQ(proxy->Info().labels.Size(), 0);
  }

  iter.Reset();
}

IterativeDMatrix::IterativeDMatrix(std::shared_ptr<EllpackPage> ellpack, MetaInfo const& info,
                                   BatchParam batch) {
  this->ellpack_ = ellpack;
  CHECK_EQ(this->info_.num_row_, 0);
  CHECK_EQ(this->info_.num_col_, 0);
  this->info_.Extend(info, true, true);
  this->info_.num_nonzero_ = info.num_nonzero_;
  CHECK_EQ(this->info_.num_row_, info.num_row_);
  this->batch_ = batch;
}

BatchSet<EllpackPage> IterativeDMatrix::GetEllpackBatches(Context const* ctx,
                                                          BatchParam const& param) {
  if (param.Initialized()) {
    detail::CheckParam(this->batch_, param);
    CHECK(!detail::RegenGHist(param, batch_)) << error::InconsistentMaxBin();
  }
  if (!ellpack_ && !ghist_) {
    LOG(FATAL) << "`QuantileDMatrix` not initialized.";
  }

  if (!ellpack_) {
    ellpack_.reset(new EllpackPage());
    if (ctx->IsCUDA()) {
      this->Info().feature_types.SetDevice(ctx->Device());
      *ellpack_->Impl() =
          EllpackPageImpl(ctx, *this->ghist_, this->Info().feature_types.ConstDeviceSpan());
    } else if (fmat_ctx_.IsCUDA()) {
      this->Info().feature_types.SetDevice(fmat_ctx_.Device());
      *ellpack_->Impl() =
          EllpackPageImpl(&fmat_ctx_, *this->ghist_, this->Info().feature_types.ConstDeviceSpan());
    } else {
      // Can happen when QDM is initialized on CPU, but a GPU version is queried by a different QDM
      // for cut reference.
      auto cuda_ctx = ctx->MakeCUDA();
      this->Info().feature_types.SetDevice(cuda_ctx.Device());
      *ellpack_->Impl() =
          EllpackPageImpl(&cuda_ctx, *this->ghist_, this->Info().feature_types.ConstDeviceSpan());
    }
  }
  CHECK(ellpack_);
  auto begin_iter = BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(ellpack_));
  return BatchSet<EllpackPage>(begin_iter);
}
}  // namespace xgboost::data
