/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#include <algorithm>  // for max
#include <numeric>    // for partial_sum
#include <vector>     // for vector

#include "../collective/allreduce.h"    // for Allreduce
#include "../common/cuda_context.cuh"   // for CUDAContext
#include "../common/cuda_rt_utils.h"    // for AllVisibleGPUs
#include "../common/cuda_rt_utils.h"    // for xgboost_NVTX_FN_RANGE
#include "../common/device_vector.cuh"  // for XGBCachingDeviceAllocator
#include "../common/hist_util.cuh"      // for AdapterDeviceSketch
#include "../common/quantile.cuh"       // for SketchContainer
#include "ellpack_page.cuh"             // for EllpackPage
#include "proxy_dmatrix.cuh"            // for Dispatch
#include "proxy_dmatrix.h"              // for DataIterProxy
#include "quantile_dmatrix.h"           // for GetCutsFromRef

namespace xgboost::data {
void GetCutsFromEllpack(EllpackPage const& page, common::HistogramCuts* cuts) {
  *cuts = page.Impl()->Cuts();
}

namespace cuda_impl {
void MakeSketches(Context const* ctx,
                  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>* iter,
                  DMatrixProxy* proxy, std::shared_ptr<DMatrix> ref, BatchParam const& p,
                  float missing, std::shared_ptr<common::HistogramCuts> cuts, MetaInfo const& info,
                  ExternalDataInfo* p_ext_info) {
  xgboost_NVTX_FN_RANGE();

  std::unique_ptr<common::SketchContainer> sketch;
  auto& ext_info = *p_ext_info;

  do {
    // We use do while here as the first batch is fetched in ctor
    CHECK_LT(ctx->Ordinal(), common::AllVisibleGPUs());
    common::SetDevice(dh::GetDevice(ctx).ordinal);
    if (ext_info.n_features == 0) {
      ext_info.n_features = data::BatchColumns(proxy);
      auto rc = collective::Allreduce(ctx, linalg::MakeVec(&ext_info.n_features, 1),
                                      collective::Op::kMax);
      SafeColl(rc);
    } else {
      CHECK_EQ(ext_info.n_features, ::xgboost::data::BatchColumns(proxy))
          << "Inconsistent number of columns.";
    }
    if (!ref) {
      if (!sketch) {
        sketch = std::make_unique<common::SketchContainer>(
            proxy->Info().feature_types, p.max_bin, ext_info.n_features, data::BatchSamples(proxy),
            dh::GetDevice(ctx));
      }
      proxy->Info().weights_.SetDevice(dh::GetDevice(ctx));
      cuda_impl::Dispatch(proxy, [&](auto const& value) {
        // Workaround empty input with CPU ctx.
        Context new_ctx;
        Context const* p_ctx;
        if (ctx->IsCUDA()) {
          p_ctx = ctx;
        } else {
          new_ctx.UpdateAllowUnknown(Args{{"device", dh::GetDevice(ctx).Name()}});
          p_ctx = &new_ctx;
        }
        common::AdapterDeviceSketch(p_ctx, value, p.max_bin, proxy->Info(), missing, sketch.get());
      });
    }
    auto batch_rows = data::BatchSamples(proxy);
    ext_info.accumulated_rows += batch_rows;
    dh::device_vector<size_t> row_counts(batch_rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(), row_counts.size());
    ext_info.row_stride =
        std::max(ext_info.row_stride, cuda_impl::Dispatch(proxy, [=](auto const& value) {
                   return GetRowCounts(value, row_counts_span, dh::GetDevice(ctx), missing);
                 }));
    ext_info.nnz += thrust::reduce(ctx->CUDACtx()->CTP(), row_counts.begin(), row_counts.end());
    ext_info.n_batches++;
    ext_info.base_rows.push_back(batch_rows);
  } while (iter->Next());
  iter->Reset();

  CHECK_GE(ext_info.n_features, 1) << "Data must has at least 1 column.";
  std::partial_sum(ext_info.base_rows.cbegin(), ext_info.base_rows.cend(),
                   ext_info.base_rows.begin());

  // Get reference
  common::SetDevice(dh::GetDevice(ctx).ordinal);
  if (!ref) {
    sketch->MakeCuts(ctx, cuts.get(), info.IsColumnSplit());
  } else {
    GetCutsFromRef(ctx, ref, ext_info.n_features, p, cuts.get());
  }
}
}  // namespace cuda_impl
}  // namespace xgboost::data
