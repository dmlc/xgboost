/**
 * Copyright 2020-2023, XGBoost contributors
 */
#include <algorithm>
#include <memory>
#include <type_traits>

#include "../common/hist_util.cuh"
#include "batch_utils.h"  // for RegenGHist
#include "device_adapter.cuh"
#include "ellpack_page.cuh"
#include "gradient_index.h"
#include "iterative_dmatrix.h"
#include "proxy_dmatrix.cuh"
#include "proxy_dmatrix.h"
#include "simple_batch_iterator.h"
#include "sparse_page_source.h"

namespace xgboost::data {
void IterativeDMatrix::InitFromCUDA(Context const* ctx, BatchParam const& p,
                                    DataIterHandle iter_handle, float missing,
                                    std::shared_ptr<DMatrix> ref) {
  // A handle passed to external iterator.
  DMatrixProxy* proxy = MakeProxy(proxy_);
  CHECK(proxy);

  // The external iterator
  auto iter =
      DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{iter_handle, reset_, next_};

  dh::XGBCachingDeviceAllocator<char> alloc;

  auto num_rows = [&]() {
    return Dispatch(proxy, [](auto const& value) { return value.NumRows(); });
  };
  auto num_cols = [&]() {
    return Dispatch(proxy, [](auto const& value) { return value.NumCols(); });
  };

  size_t row_stride = 0;
  size_t nnz = 0;
  // Sketch for all batches.
  std::vector<common::SketchContainer> sketch_containers;
  size_t batches = 0;
  size_t accumulated_rows = 0;
  bst_feature_t cols = 0;

  int32_t current_device;
  dh::safe_cuda(cudaGetDevice(&current_device));
  auto get_device = [&]() -> int32_t {
    std::int32_t d = (ctx->gpu_id == Context::kCpuId) ? current_device : ctx->gpu_id;
    CHECK_NE(d, Context::kCpuId);
    return d;
  };

  /**
   * Generate quantiles
   */
  common::HistogramCuts cuts;
  do {
    // We use do while here as the first batch is fetched in ctor
    // ctx_.gpu_id = proxy->DeviceIdx();
    CHECK_LT(ctx->gpu_id, common::AllVisibleGPUs());
    dh::safe_cuda(cudaSetDevice(get_device()));
    if (cols == 0) {
      cols = num_cols();
      collective::Allreduce<collective::Operation::kMax>(&cols, 1);
      this->info_.num_col_ = cols;
    } else {
      CHECK_EQ(cols, num_cols()) << "Inconsistent number of columns.";
    }
    if (!ref) {
      sketch_containers.emplace_back(proxy->Info().feature_types, p.max_bin, cols, num_rows(),
                                     get_device());
      auto* p_sketch = &sketch_containers.back();
      proxy->Info().weights_.SetDevice(get_device());
      Dispatch(proxy, [&](auto const& value) {
        common::AdapterDeviceSketch(value, p.max_bin, proxy->Info(), missing, p_sketch);
      });
    }
    auto batch_rows = num_rows();
    accumulated_rows += batch_rows;
    dh::caching_device_vector<size_t> row_counts(batch_rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(), row_counts.size());
    row_stride = std::max(row_stride, Dispatch(proxy, [=](auto const& value) {
                            return GetRowCounts(value, row_counts_span, get_device(), missing);
                          }));
    nnz += thrust::reduce(thrust::cuda::par(alloc), row_counts.begin(), row_counts.end());
    batches++;
  } while (iter.Next());
  iter.Reset();

  auto n_features = cols;
  CHECK_GE(n_features, 1) << "Data must has at least 1 column.";

  dh::safe_cuda(cudaSetDevice(get_device()));
  if (!ref) {
    HostDeviceVector<FeatureType> ft;
    common::SketchContainer final_sketch(
        sketch_containers.empty() ? ft : sketch_containers.front().FeatureTypes(), p.max_bin, cols,
        accumulated_rows, get_device());
    for (auto const& sketch : sketch_containers) {
      final_sketch.Merge(sketch.ColumnsPtr(), sketch.Data());
      final_sketch.FixError();
    }
    sketch_containers.clear();
    sketch_containers.shrink_to_fit();

    final_sketch.MakeCuts(&cuts);
  } else {
    GetCutsFromRef(ctx, ref, Info().num_col_, p, &cuts);
  }

  this->info_.num_row_ = accumulated_rows;
  this->info_.num_nonzero_ = nnz;

  auto init_page = [this, &proxy, &cuts, row_stride, accumulated_rows, get_device]() {
    if (!ellpack_) {
      // Should be put inside the while loop to protect against empty batch.  In
      // that case device id is invalid.
      ellpack_.reset(new EllpackPage);
      *(ellpack_->Impl()) =
          EllpackPageImpl(get_device(), cuts, this->IsDense(), row_stride, accumulated_rows);
    }
  };

  /**
   * Generate gradient index.
   */
  size_t offset = 0;
  iter.Reset();
  size_t n_batches_for_verification = 0;
  while (iter.Next()) {
    init_page();
    dh::safe_cuda(cudaSetDevice(get_device()));
    auto rows = num_rows();
    dh::caching_device_vector<size_t> row_counts(rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(), row_counts.size());
    Dispatch(proxy, [=](auto const& value) {
      return GetRowCounts(value, row_counts_span, get_device(), missing);
    });
    auto is_dense = this->IsDense();

    proxy->Info().feature_types.SetDevice(get_device());
    auto d_feature_types = proxy->Info().feature_types.ConstDeviceSpan();
    auto new_impl = Dispatch(proxy, [&](auto const& value) {
      return EllpackPageImpl(value, missing, get_device(), is_dense, row_counts_span,
                             d_feature_types, row_stride, rows, cuts);
    });
    size_t num_elements = ellpack_->Impl()->Copy(get_device(), &new_impl, offset);
    offset += num_elements;

    proxy->Info().num_row_ = num_rows();
    proxy->Info().num_col_ = cols;
    if (batches != 1) {
      this->info_.Extend(std::move(proxy->Info()), false, true);
    }
    n_batches_for_verification++;
  }
  CHECK_EQ(batches, n_batches_for_verification)
      << "Different number of batches returned between 2 iterations";

  if (batches == 1) {
    this->info_ = std::move(proxy->Info());
    this->info_.num_nonzero_ = nnz;
    CHECK_EQ(proxy->Info().labels.Size(), 0);
  }

  iter.Reset();
  // Synchronise worker columns
  info_.SynchronizeNumberOfColumns();
}

BatchSet<EllpackPage> IterativeDMatrix::GetEllpackBatches(Context const* ctx,
                                                          BatchParam const& param) {
  if (param.Initialized()) {
    CheckParam(param);
    CHECK(!detail::RegenGHist(param, batch_)) << error::InconsistentMaxBin();
  }
  if (!ellpack_ && !ghist_) {
    LOG(FATAL) << "`QuantileDMatrix` not initialized.";
  }

  if (!ellpack_) {
    ellpack_.reset(new EllpackPage());
    if (ctx->IsCUDA()) {
      this->Info().feature_types.SetDevice(ctx->gpu_id);
      *ellpack_->Impl() =
          EllpackPageImpl(ctx, *this->ghist_, this->Info().feature_types.ConstDeviceSpan());
    } else if (fmat_ctx_.IsCUDA()) {
      this->Info().feature_types.SetDevice(fmat_ctx_.gpu_id);
      *ellpack_->Impl() =
          EllpackPageImpl(&fmat_ctx_, *this->ghist_, this->Info().feature_types.ConstDeviceSpan());
    } else {
      // Can happen when QDM is initialized on CPU, but a GPU version is queried by a different QDM
      // for cut reference.
      auto cuda_ctx = ctx->MakeCUDA();
      this->Info().feature_types.SetDevice(cuda_ctx.gpu_id);
      *ellpack_->Impl() =
          EllpackPageImpl(&cuda_ctx, *this->ghist_, this->Info().feature_types.ConstDeviceSpan());
    }
  }
  CHECK(ellpack_);
  auto begin_iter = BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(ellpack_));
  return BatchSet<EllpackPage>(begin_iter);
}

void GetCutsFromEllpack(EllpackPage const& page, common::HistogramCuts* cuts) {
  *cuts = page.Impl()->Cuts();
}
}  // namespace xgboost::data
