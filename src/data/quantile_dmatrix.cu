/**
 * Copyright 2020-2025, XGBoost Contributors
 */
#include <algorithm>  // for max
#include <limits>     // for numeric_limits
#include <numeric>    // for partial_sum
#include <vector>     // for vector

#include "../collective/allreduce.h"         // for Allreduce
#include "../collective/communicator-inl.h"  // for IsDistributed
#include "../common/cuda_context.cuh"        // for CUDAContext
#include "../common/cuda_rt_utils.h"         // for AllVisibleGPUs
#include "../common/device_vector.cuh"       // for XGBCachingDeviceAllocator
#include "../common/error_msg.h"             // for InconsistentCategories
#include "../common/hist_util.cuh"           // for AdapterDeviceSketch
#include "../common/nvtx_utils.h"            // for xgboost_NVTX_FN_RANGE
#include "../common/quantile.cuh"            // for SketchContainer
#include "cat_container.h"                   // for CatContainer
#include "cat_container_hash.cuh"            // for HashCatDeviceContent
#include "cat_container_hash.h"              // for CatContentDigest
#include "ellpack_page.cuh"                  // for EllpackPage
#include "proxy_dmatrix.cuh"                 // for DispatchAny
#include "proxy_dmatrix.h"                   // for DataIterProxy
#include "quantile_dmatrix.h"                // for GetCutsFromRef

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
  // Lazy because we need the `n_features`.
  std::unique_ptr<common::SketchContainer> sketch;
  auto& ext_info = *p_ext_info;

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
    // We use do while here as the first batch has been fetched in the ctor
    CHECK_LT(ctx->Ordinal(), curt::AllVisibleGPUs());
    auto device = dh::GetDevice(ctx);
    curt::SetDevice(device.ordinal);
    auto cats = cuda_impl::BatchCats(proxy);
    if (ext_info.n_features == 0) {
      ext_info.n_features = data::BatchColumns(proxy);
      // kMax(n_features) before the categorical digest -- matches cpu_impl::GetDataShape order
      auto rc = collective::Allreduce(ctx, linalg::MakeVec(&ext_info.n_features, 1),
                                      collective::Op::kMax);
      SafeColl(rc);
      // build the per-batch dictionary unconditionally; the predictor's Recode flow
      // requires ext_info.cats to be the proxy's own dictionary so MakeCatAccessor can
      // map predict-side codes against the (separate) ref-side dictionary held in the
      // model. Aliasing ext_info.cats to ref->Info().CatsShared() collapses the two
      // dictionaries and breaks Recode (test_ordinal.py::test_recode_dmatrix_predict
      // and ::test_cat_shap surface this immediately).
      ext_info.cats =
          std::make_shared<CatContainer>(p_ctx, cats, ::xgboost::data::BatchCatsIsRef(proxy));
      if (collective::IsDistributed()) {
        bool const has_cats =
            ref && ext_info.cats && ext_info.cats->HasCategorical();
        if (AllreduceRefAgreement(ctx, ref != nullptr, has_cats)) {
          // hash the ref dictionary directly; on a CUDA-side ref keep the fold on-device
          auto const ref_cats = ref->Info().CatsShared();
          CatContentDigest digest;
          if (ref_cats->DeviceCanRead()) {
            digest = HashCatDeviceContent(ctx, ref_cats->DeviceView(ctx));
          } else {
            digest = HashCatHostContent(ref_cats->HostView());
          }
          AllreduceDigestAndCheck(ctx, digest);
        }
      }
    } else {
      CHECK_EQ(ext_info.n_features, data::BatchColumns(proxy)) << "Inconsistent number of columns.";
      CHECK_EQ(cats.n_total_cats, ext_info.cats->NumCatsTotal()) << error::InconsistentCategories();
    }

    auto batch_rows = data::BatchSamples(proxy);
    ext_info.accumulated_rows += batch_rows;
    /**
     * Handle sketching.
     */
    if (!ref) {
      if (!sketch) {
        sketch = std::make_unique<common::SketchContainer>(proxy->Info().feature_types, p.max_bin,
                                                           ext_info.n_features, dh::GetDevice(ctx));
      }
      proxy->Info().weights_.SetDevice(dh::GetDevice(ctx));
      DispatchAny(proxy, [&](auto const& value) {
        common::AdapterDeviceSketch(p_ctx, value, p.max_bin, proxy->Info(), missing, sketch.get());
      });
      LOG(DEBUG) << "Total capacity:" << common::HumanMemUnit(sketch->MemCapacityBytes());
    }

    /**
     * Rest of the data shape.
     */
    dh::device_vector<size_t> row_counts(batch_rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(), row_counts.size());
    ext_info.row_stride =
        std::max(ext_info.row_stride, DispatchAny(proxy, [=](auto const& value) {
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
    if (!sketch) {
      // Empty local input can happen in distributed settings.
      sketch = std::make_unique<common::SketchContainer>(proxy->Info().feature_types, p.max_bin,
                                                         ext_info.n_features, dh::GetDevice(ctx));
    }
    *cuts = sketch->MakeCuts(ctx, info.IsColumnSplit());
    sketch.reset();
  } else {
    GetCutsFromRef(ctx, ref, ext_info.n_features, p, cuts.get());
  }

  ctx->CUDACtx()->Stream().Sync();
}
}  // namespace cuda_impl
}  // namespace xgboost::data
