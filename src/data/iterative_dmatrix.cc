/**
 * Copyright 2022-2023, XGBoost contributors
 */
#include "iterative_dmatrix.h"

#include <algorithm>    // for copy
#include <cstddef>      // for size_t
#include <memory>       // for shared_ptr
#include <type_traits>  // for underlying_type_t
#include <vector>       // for vector

#include "../collective/communicator-inl.h"
#include "../common/categorical.h"  // common::IsCat
#include "../common/column_matrix.h"
#include "../tree/param.h"          // FIXME(jiamingy): Find a better way to share this parameter.
#include "batch_utils.h"            // for RegenGHist
#include "gradient_index.h"
#include "proxy_dmatrix.h"
#include "simple_batch_iterator.h"
#include "xgboost/data.h"  // for FeatureType, DMatrix
#include "xgboost/logging.h"

namespace xgboost::data {
IterativeDMatrix::IterativeDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy,
                                   std::shared_ptr<DMatrix> ref, DataIterResetCallback* reset,
                                   XGDMatrixCallbackNext* next, float missing, int nthread,
                                   bst_bin_t max_bin)
    : proxy_{proxy}, reset_{reset}, next_{next} {
  // fetch the first batch
  auto iter =
      DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{iter_handle, reset_, next_};
  iter.Reset();
  bool valid = iter.Next();
  CHECK(valid) << "Iterative DMatrix must have at least 1 batch.";

  auto d = MakeProxy(proxy_)->DeviceIdx();

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"nthread", std::to_string(nthread)}, {"gpu_id", std::to_string(d)}});
  // hardcoded parameter.
  BatchParam p{max_bin, tree::TrainParam::DftSparseThreshold()};

  if (ctx.IsCPU()) {
    this->InitFromCPU(&ctx, p, iter_handle, missing, ref);
  } else {
    this->InitFromCUDA(&ctx, p, iter_handle, missing, ref);
  }

  this->fmat_ctx_ = ctx;
  this->batch_ = p;
}

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
    if (ctx->IsCPU()) {
      csr();
    } else {
      ellpack();
    }
  } else if (ref->PageExists<GHistIndexMatrix>()) {
    csr();
  } else if (ref->PageExists<EllpackPage>()) {
    ellpack();
  } else {
    // None exist
    if (ctx->IsCPU()) {
      csr();
    } else {
      ellpack();
    }
  }
  CHECK_EQ(ref->Info().num_col_, n_features)
      << "Invalid ref DMatrix, different number of features.";
}

namespace {
// Synchronize feature type in case of empty DMatrix
void SyncFeatureType(std::vector<FeatureType>* p_h_ft) {
  if (!collective::IsDistributed()) {
    return;
  }
  auto& h_ft = *p_h_ft;
  auto n_ft = h_ft.size();
  collective::Allreduce<collective::Operation::kMax>(&n_ft, 1);
  if (!h_ft.empty()) {
    // Check correct size if this is not an empty DMatrix.
    CHECK_EQ(h_ft.size(), n_ft);
  }
  if (n_ft > 0) {
    h_ft.resize(n_ft);
    auto ptr = reinterpret_cast<std::underlying_type_t<FeatureType>*>(h_ft.data());
    collective::Allreduce<collective::Operation::kMax>(ptr, h_ft.size());
  }
}
}  // anonymous namespace

void IterativeDMatrix::InitFromCPU(Context const* ctx, BatchParam const& p,
                                   DataIterHandle iter_handle, float missing,
                                   std::shared_ptr<DMatrix> ref) {
  DMatrixProxy* proxy = MakeProxy(proxy_);
  CHECK(proxy);

  // The external iterator
  auto iter =
      DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{iter_handle, reset_, next_};
  common::HistogramCuts cuts;

  auto num_rows = [&]() {
    return HostAdapterDispatch(proxy, [](auto const& value) { return value.Size(); });
  };
  auto num_cols = [&]() {
    return HostAdapterDispatch(proxy, [](auto const& value) { return value.NumCols(); });
  };

  std::vector<std::size_t> column_sizes;
  auto const is_valid = data::IsValidFunctor{missing};
  auto nnz_cnt = [&]() {
    return HostAdapterDispatch(proxy, [&](auto const& value) {
      size_t n_threads = ctx->Threads();
      size_t n_features = column_sizes.size();
      linalg::Tensor<std::size_t, 2> column_sizes_tloc({n_threads, n_features}, Context::kCpuId);
      column_sizes_tloc.Data()->Fill(0ul);
      auto view = column_sizes_tloc.HostView();
      common::ParallelFor(value.Size(), n_threads, common::Sched::Static(256), [&](auto i) {
        auto const& line = value.GetLine(i);
        for (size_t j = 0; j < line.Size(); ++j) {
          data::COOTuple const& elem = line.GetElement(j);
          if (is_valid(elem)) {
            view(omp_get_thread_num(), elem.column_idx)++;
          }
        }
      });
      auto ptr = column_sizes_tloc.Data()->HostPointer();
      auto result = std::accumulate(ptr, ptr + column_sizes_tloc.Size(), static_cast<size_t>(0));
      for (size_t tidx = 0; tidx < n_threads; ++tidx) {
        for (size_t fidx = 0; fidx < n_features; ++fidx) {
          column_sizes[fidx] += view(tidx, fidx);
        }
      }
      return result;
    });
  };

  std::uint64_t n_features = 0;
  std::size_t n_batches = 0;
  std::uint64_t accumulated_rows{0};
  std::uint64_t nnz{0};

  /**
   * CPU impl needs an additional loop for accumulating the column size.
   */
  std::unique_ptr<common::HostSketchContainer> p_sketch;
  std::vector<size_t> batch_nnz;
  do {
    // We use do while here as the first batch is fetched in ctor
    if (n_features == 0) {
      n_features = num_cols();
      collective::Allreduce<collective::Operation::kMax>(&n_features, 1);
      column_sizes.clear();
      column_sizes.resize(n_features, 0);
      info_.num_col_ = n_features;
    } else {
      CHECK_EQ(n_features, num_cols()) << "Inconsistent number of columns.";
    }
    size_t batch_size = num_rows();
    batch_nnz.push_back(nnz_cnt());
    nnz += batch_nnz.back();
    accumulated_rows += batch_size;
    n_batches++;
  } while (iter.Next());
  iter.Reset();

  // From here on Info() has the correct data shape
  Info().num_row_ = accumulated_rows;
  Info().num_nonzero_ = nnz;
  Info().SynchronizeNumberOfColumns();
  CHECK(std::none_of(column_sizes.cbegin(), column_sizes.cend(), [&](auto f) {
    return f > accumulated_rows;
  })) << "Something went wrong during iteration.";

  CHECK_GE(n_features, 1) << "Data must has at least 1 column.";

  /**
   * Generate quantiles
   */
  accumulated_rows = 0;
  std::vector<FeatureType> h_ft;
  if (ref) {
    GetCutsFromRef(ctx, ref, Info().num_col_, p, &cuts);
    h_ft = ref->Info().feature_types.HostVector();
  } else {
    size_t i = 0;
    while (iter.Next()) {
      if (!p_sketch) {
        h_ft = proxy->Info().feature_types.ConstHostVector();
        SyncFeatureType(&h_ft);
        p_sketch.reset(new common::HostSketchContainer{ctx, p.max_bin, h_ft, column_sizes,
                                                       !proxy->Info().group_ptr_.empty()});
      }
      HostAdapterDispatch(proxy, [&](auto const& batch) {
        proxy->Info().num_nonzero_ = batch_nnz[i];
        // We don't need base row idx here as Info is from proxy and the number of rows in
        // it is consistent with data batch.
        p_sketch->PushAdapterBatch(batch, 0, proxy->Info(), missing);
      });
      accumulated_rows += num_rows();
      ++i;
    }
    iter.Reset();
    CHECK_EQ(accumulated_rows, Info().num_row_);

    CHECK(p_sketch);
    p_sketch->MakeCuts(Info(), &cuts);
  }
  if (!h_ft.empty()) {
    CHECK_EQ(h_ft.size(), n_features);
  }

  /**
   * Generate gradient index.
   */
  this->ghist_ = std::make_unique<GHistIndexMatrix>(Info(), std::move(cuts), p.max_bin);
  size_t rbegin = 0;
  size_t prev_sum = 0;
  size_t i = 0;
  while (iter.Next()) {
    HostAdapterDispatch(proxy, [&](auto const& batch) {
      proxy->Info().num_nonzero_ = batch_nnz[i];
      this->ghist_->PushAdapterBatch(ctx, rbegin, prev_sum, batch, missing, h_ft, p.sparse_thresh,
                                     Info().num_row_);
    });
    if (n_batches != 1) {
      this->info_.Extend(std::move(proxy->Info()), false, true);
    }
    size_t batch_size = num_rows();
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
  accumulated_rows = 0;
  while (iter.Next()) {
    HostAdapterDispatch(proxy, [&](auto const& batch) {
      this->ghist_->PushAdapterBatchColumns(ctx, batch, missing, accumulated_rows);
    });
    accumulated_rows += num_rows();
  }
  iter.Reset();
  CHECK_EQ(accumulated_rows, Info().num_row_);

  if (n_batches == 1) {
    this->info_ = std::move(proxy->Info());
    this->info_.num_nonzero_ = nnz;
    this->info_.num_col_ = n_features;  // proxy might be empty.
    CHECK_EQ(proxy->Info().labels.Size(), 0);
  }

  Info().feature_types.HostVector() = h_ft;
}

BatchSet<GHistIndexMatrix> IterativeDMatrix::GetGradientIndex(Context const* ctx,
                                                              BatchParam const& param) {
  if (param.Initialized()) {
    CheckParam(param);
    CHECK(!detail::RegenGHist(param, batch_)) << error::InconsistentMaxBin();
  }
  if (!ellpack_ && !ghist_) {
    LOG(FATAL) << "`QuantileDMatrix` not initialized.";
  }

  if (!ghist_) {
    if (ctx->IsCPU()) {
      ghist_ = std::make_shared<GHistIndexMatrix>(ctx, Info(), *ellpack_, param);
    } else if (fmat_ctx_.IsCPU()) {
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
inline void IterativeDMatrix::InitFromCUDA(Context const*, BatchParam const&, DataIterHandle, float,
                                           std::shared_ptr<DMatrix>) {
  // silent the warning about unused variables.
  (void)(proxy_);
  (void)(reset_);
  (void)(next_);
  common::AssertGPUSupport();
}

inline BatchSet<EllpackPage> IterativeDMatrix::GetEllpackBatches(Context const* ctx,
                                                                 BatchParam const& param) {
  common::AssertGPUSupport();
  auto begin_iter = BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(ellpack_));
  return BatchSet<EllpackPage>(BatchIterator<EllpackPage>(begin_iter));
}

inline void GetCutsFromEllpack(EllpackPage const&, common::HistogramCuts*) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data
