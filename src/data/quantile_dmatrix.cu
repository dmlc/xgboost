/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#include <algorithm>  // for max
#include <numeric>    // for partial_sum
#include <vector>     // for vector

#include "../collective/allreduce.h"    // for Allreduce
#include "../common/cuda_rt_utils.h"    // for AllVisibleGPUs
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
  dh::XGBCachingDeviceAllocator<char> alloc;
  std::vector<common::SketchContainer> sketch_containers;
  auto& ext_info = *p_ext_info;

  do {
    // We use do while here as the first batch is fetched in ctor
    CHECK_LT(ctx->Ordinal(), common::AllVisibleGPUs());
    dh::safe_cuda(cudaSetDevice(dh::GetDevice(ctx).ordinal));
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
      sketch_containers.emplace_back(proxy->Info().feature_types, p.max_bin, ext_info.n_features,
                                     data::BatchSamples(proxy), dh::GetDevice(ctx));
      auto* p_sketch = &sketch_containers.back();
      proxy->Info().weights_.SetDevice(dh::GetDevice(ctx));
      cuda_impl::Dispatch(proxy, [&](auto const& value) {
        common::AdapterDeviceSketch(value, p.max_bin, proxy->Info(), missing, p_sketch);
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
    ext_info.nnz += thrust::reduce(thrust::cuda::par(alloc), row_counts.begin(), row_counts.end());
    ext_info.n_batches++;
    ext_info.base_rows.push_back(batch_rows);
  } while (iter->Next());
  iter->Reset();

  CHECK_GE(ext_info.n_features, 1) << "Data must has at least 1 column.";
  std::partial_sum(ext_info.base_rows.cbegin(), ext_info.base_rows.cend(),
                   ext_info.base_rows.begin());

  // Get reference
  dh::safe_cuda(cudaSetDevice(dh::GetDevice(ctx).ordinal));
  if (!ref) {
    HostDeviceVector<FeatureType> ft;
    common::SketchContainer final_sketch(
        sketch_containers.empty() ? ft : sketch_containers.front().FeatureTypes(), p.max_bin,
        ext_info.n_features, ext_info.accumulated_rows, dh::GetDevice(ctx));
    for (auto const& sketch : sketch_containers) {
      final_sketch.Merge(sketch.ColumnsPtr(), sketch.Data());
      final_sketch.FixError();
    }
    sketch_containers.clear();
    sketch_containers.shrink_to_fit();

    final_sketch.MakeCuts(ctx, cuts.get(), info.IsColumnSplit());
  } else {
    GetCutsFromRef(ctx, ref, ext_info.n_features, p, cuts.get());
  }
}
}  // namespace cuda_impl
}  // namespace xgboost::data
