/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#include <algorithm>  // for max
#include <limits>     // for numeric_limits
#include <numeric>    // for partial_sum
#include <utility>    // for pair
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
                  std::int64_t max_quantile_blocks, ExternalDataInfo* p_ext_info) {
  xgboost_NVTX_FN_RANGE();
  /**
   * A variant of: A Fast Algorithm for Approximate Quantiles in High Speed Data Streams
   *
   * The original algorithm was designed for CPU where input is a stream with individual
   * elements. For GPU, we process the data in batches. As a result, the implementation
   * here simply uses the user input batch as the basic unit of sketching blocks. The
   * number of blocks per-level grows exponentially.
   */
  std::vector<std::pair<std::unique_ptr<common::SketchContainer>, bst_idx_t>> sketches;
  auto& ext_info = *p_ext_info;

  auto lazy_init_sketch = [&] {
    // Lazy because we need the `n_features`.
    sketches.emplace_back(std::make_unique<common::SketchContainer>(
                              proxy->Info().feature_types, p.max_bin, ext_info.n_features,
                              data::BatchSamples(proxy), dh::GetDevice(ctx)),
                          0);
  };
  auto total_capacity = [&] {
    bst_idx_t n_bytes = 0;
    for (auto const& sk : sketches) {
      n_bytes += sk.first->MemCapacityBytes();
    }
    return n_bytes;
  };

  // Workaround empty input with CPU ctx.
  Context new_ctx;
  Context const* p_ctx;
  if (ctx->IsCUDA()) {
    p_ctx = ctx;
  } else {
    new_ctx.UpdateAllowUnknown(Args{{"device", dh::GetDevice(ctx).Name()}});
    p_ctx = &new_ctx;
  }

  do {
    /**
     * Get the data shape.
     */
    // We use do while here as the first batch is fetched in ctor
    CHECK_LT(ctx->Ordinal(), curt::AllVisibleGPUs());
    curt::SetDevice(dh::GetDevice(ctx).ordinal);
    if (ext_info.n_features == 0) {
      ext_info.n_features = data::BatchColumns(proxy);
      auto rc = collective::Allreduce(ctx, linalg::MakeVec(&ext_info.n_features, 1),
                                      collective::Op::kMax);
      SafeColl(rc);
    } else {
      CHECK_EQ(ext_info.n_features, data::BatchColumns(proxy))
          << "Inconsistent number of columns.";
    }

    auto batch_rows = data::BatchSamples(proxy);
    ext_info.accumulated_rows += batch_rows;

    /**
     * Handle sketching.
     */
    if (!ref) {
      CHECK_LE(max_quantile_blocks, std::numeric_limits<bst_idx_t>::max());
      CHECK_GT(max_quantile_blocks, 0) << "`max_quantile_blocks` must be greater than 0.";
      if (sketches.empty()) {
        lazy_init_sketch();
      }
      if (sketches.back().second > (1ul << (sketches.size() - 1)) ||
          sketches.back().second == static_cast<bst_idx_t>(max_quantile_blocks)) {
        // Cut the sub-stream.
        auto n_cuts_per_feat =
            common::detail::RequiredSampleCutsPerColumn(p.max_bin, ext_info.accumulated_rows);
        // Prune to a single block
        sketches.back().first->Prune(p_ctx, n_cuts_per_feat);
        sketches.back().first->ShrinkToFit();

        sketches.back().second = 1;
        lazy_init_sketch();  // Add a new level.
      }
      proxy->Info().weights_.SetDevice(dh::GetDevice(ctx));
      Dispatch(proxy, [&](auto const& value) {
        common::AdapterDeviceSketch(p_ctx, value, p.max_bin, proxy->Info(), missing,
                                    sketches.back().first.get());
        sketches.back().second++;
      });
      LOG(DEBUG) << "Total capacity:" << common::HumanMemUnit(total_capacity());
    }

    /**
     * Rest of the data shape.
     */
    dh::device_vector<size_t> row_counts(batch_rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(), row_counts.size());
    ext_info.row_stride =
        std::max(ext_info.row_stride, Dispatch(proxy, [=](auto const& value) {
                   return GetRowCounts(ctx, value, row_counts_span, dh::GetDevice(ctx), missing);
                 }));
    ext_info.nnz += thrust::reduce(ctx->CUDACtx()->CTP(), row_counts.begin(), row_counts.end());
    ext_info.n_batches++;
    ext_info.base_rowids.push_back(batch_rows);
  } while (iter->Next());
  iter->Reset();

  CHECK_GE(ext_info.n_features, 1) << "Data must has at least 1 column.";
  std::partial_sum(ext_info.base_rowids.cbegin(), ext_info.base_rowids.cend(),
                   ext_info.base_rowids.begin());

  // Get reference
  curt::SetDevice(dh::GetDevice(ctx).ordinal);
  if (!ref) {
    HostDeviceVector<FeatureType> ft;
    common::SketchContainer final_sketch(
        sketches.empty() ? ft : sketches.front().first->FeatureTypes(), p.max_bin,
        ext_info.n_features, ext_info.accumulated_rows, dh::GetDevice(ctx));
    // Reverse order since the last container might contain summary that's not yet pruned.
    for (auto it = sketches.crbegin(); it != sketches.crend(); ++it) {
      auto& sketch = *it;

      CHECK_GE(sketch.second, 1);
      if (sketch.second > 1) {
        sketch.first->Prune(p_ctx, common::detail::RequiredSampleCutsPerColumn(
                                       p.max_bin, ext_info.accumulated_rows));
        sketch.first->ShrinkToFit();
      }
      final_sketch.Merge(p_ctx, sketch.first->ColumnsPtr(), sketch.first->Data());
      final_sketch.FixError();
    }

    sketches.clear();
    sketches.shrink_to_fit();

    final_sketch.MakeCuts(ctx, cuts.get(), info.IsColumnSplit());
  } else {
    GetCutsFromRef(ctx, ref, ext_info.n_features, p, cuts.get());
  }

  ctx->CUDACtx()->Stream().Sync();
}
}  // namespace cuda_impl
}  // namespace xgboost::data
