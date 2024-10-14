/**
 * Copyright 2024, XGBoost Contributors
 */
#include "quantile_dmatrix.h"

#include <numeric>  // for accumulate

#include "../collective/allreduce.h"         // for Allreduce
#include "../collective/communicator-inl.h"  // for IsDistributed
#include "../common/threading_utils.h"       // for ParallelFor
#include "gradient_index.h"                  // for GHistIndexMatrix
#include "xgboost/collective/result.h"       // for SafeColl
#include "xgboost/linalg.h"                  // for Tensor

namespace xgboost::data {
void GetCutsFromRef(Context const* ctx, std::shared_ptr<DMatrix> ref, bst_feature_t n_features,
                    BatchParam p, common::HistogramCuts* p_cuts) {
  CHECK(ref);
  CHECK(p_cuts);
  p.forbid_regen = true;
  // Fetch cuts from GIDX
  auto csr = [&] {
    for (auto const& page : ref->GetBatches<GHistIndexMatrix>(ctx, p)) {
      *p_cuts = page.cut;
      break;
    }
  };
  // Fetch cuts from Ellpack.
  auto ellpack = [&] {
    for (auto const& page : ref->GetBatches<EllpackPage>(ctx, p)) {
      GetCutsFromEllpack(page, p_cuts);
      break;
    }
  };

  if (ref->PageExists<GHistIndexMatrix>() && ref->PageExists<EllpackPage>()) {
    // Both exists
    if (ctx->IsCUDA()) {
      ellpack();
    } else {
      csr();
    }
  } else if (ref->PageExists<GHistIndexMatrix>()) {
    csr();
  } else if (ref->PageExists<EllpackPage>()) {
    ellpack();
  } else {
    // None exist
    if (ctx->IsCUDA()) {
      ellpack();
    } else {
      csr();
    }
  }
  CHECK_EQ(ref->Info().num_col_, n_features)
      << "Invalid ref DMatrix, different number of features.";
}

#if !defined(XGBOOST_USE_CUDA)
void GetCutsFromEllpack(EllpackPage const&, common::HistogramCuts*) {
  common::AssertGPUSupport();
}
#endif

namespace cpu_impl {
// Synchronize feature type in case of empty DMatrix
void SyncFeatureType(Context const* ctx, std::vector<FeatureType>* p_h_ft) {
  if (!collective::IsDistributed()) {
    return;
  }
  auto& h_ft = *p_h_ft;
  bst_idx_t n_ft = h_ft.size();
  collective::SafeColl(collective::Allreduce(ctx, &n_ft, collective::Op::kMax));
  if (!h_ft.empty()) {
    // Check correct size if this is not an empty DMatrix.
    CHECK_EQ(h_ft.size(), n_ft);
  }
  if (n_ft > 0) {
    h_ft.resize(n_ft);
    auto ptr = reinterpret_cast<std::underlying_type_t<FeatureType>*>(h_ft.data());
    collective::SafeColl(
        collective::Allreduce(ctx, linalg::MakeVec(ptr, h_ft.size()), collective::Op::kMax));
  }
}

void GetDataShape(Context const* ctx, DMatrixProxy* proxy,
                  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> iter, float missing,
                  ExternalDataInfo* p_info) {
  auto& info = *p_info;

  auto const is_valid = data::IsValidFunctor{missing};
  auto nnz_cnt = [&]() {
    return HostAdapterDispatch(proxy, [&](auto const& value) {
      bst_idx_t n_threads = ctx->Threads();
      bst_idx_t n_features = info.column_sizes.size();
      linalg::Tensor<bst_idx_t, 2> column_sizes_tloc({n_threads, n_features}, DeviceOrd::CPU());
      column_sizes_tloc.Data()->Fill(0ul);
      auto view = column_sizes_tloc.HostView();
      common::ParallelFor(value.Size(), n_threads, common::Sched::Static(256), [&](auto i) {
        auto const& line = value.GetLine(i);
        for (bst_idx_t j = 0; j < line.Size(); ++j) {
          data::COOTuple const& elem = line.GetElement(j);
          if (is_valid(elem)) {
            view(omp_get_thread_num(), elem.column_idx)++;
          }
        }
      });
      auto ptr = column_sizes_tloc.Data()->HostPointer();
      auto result = std::accumulate(ptr, ptr + column_sizes_tloc.Size(), static_cast<bst_idx_t>(0));
      for (bst_idx_t tidx = 0; tidx < n_threads; ++tidx) {
        for (bst_idx_t fidx = 0; fidx < n_features; ++fidx) {
          info.column_sizes[fidx] += view(tidx, fidx);
        }
      }
      return result;
    });
  };

  /**
   * CPU impl needs an additional loop for accumulating the column size.
   */
  do {
    // We use do while here as the first batch is fetched in ctor
    if (info.n_features == 0) {
      info.n_features = BatchColumns(proxy);
      collective::SafeColl(collective::Allreduce(ctx, &info.n_features, collective::Op::kMax));
      info.column_sizes.clear();
      info.column_sizes.resize(info.n_features, 0);
    } else {
      CHECK_EQ(info.n_features, BatchColumns(proxy)) << "Inconsistent number of columns.";
    }
    bst_idx_t batch_size = BatchSamples(proxy);
    info.batch_nnz.push_back(nnz_cnt());
    info.base_rowids.push_back(batch_size);
    info.nnz += info.batch_nnz.back();
    info.accumulated_rows += batch_size;
    info.n_batches++;
  } while (iter.Next());
  iter.Reset();

  std::partial_sum(info.base_rowids.cbegin(), info.base_rowids.cend(), info.base_rowids.begin());
}

void MakeSketches(Context const* ctx,
                  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>* iter,
                  DMatrixProxy* proxy, std::shared_ptr<DMatrix> ref, float missing,
                  common::HistogramCuts* cuts, BatchParam const& p, MetaInfo const& info,
                  ExternalDataInfo const& ext_info, std::vector<FeatureType>* p_h_ft) {
  std::unique_ptr<common::HostSketchContainer> p_sketch;
  auto& h_ft = *p_h_ft;
  bst_idx_t accumulated_rows = 0;
  if (ref) {
    GetCutsFromRef(ctx, ref, info.num_col_, p, cuts);
    h_ft = ref->Info().feature_types.HostVector();
  } else {
    size_t i = 0;
    while (iter->Next()) {
      if (!p_sketch) {
        h_ft = proxy->Info().feature_types.ConstHostVector();
        cpu_impl::SyncFeatureType(ctx, &h_ft);
        p_sketch = std::make_unique<common::HostSketchContainer>(
            ctx, p.max_bin, h_ft, ext_info.column_sizes, !proxy->Info().group_ptr_.empty());
      }
      HostAdapterDispatch(proxy, [&](auto const& batch) {
        proxy->Info().num_nonzero_ = ext_info.batch_nnz[i];
        // We don't need base row idx here as Info is from proxy and the number of rows in
        // it is consistent with data batch.
        p_sketch->PushAdapterBatch(batch, 0, proxy->Info(), missing);
      });
      accumulated_rows += BatchSamples(proxy);
      ++i;
    }
    iter->Reset();
    CHECK_EQ(accumulated_rows, info.num_row_);

    CHECK(p_sketch);
    p_sketch->MakeCuts(ctx, info, cuts);
  }

  if (!h_ft.empty()) {
    CHECK_EQ(h_ft.size(), ext_info.n_features);
  }
}
}  // namespace cpu_impl
}  // namespace xgboost::data
