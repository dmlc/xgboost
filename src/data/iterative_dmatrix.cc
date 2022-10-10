/*!
 * Copyright 2022 XGBoost contributors
 */
#include "iterative_dmatrix.h"

#include <algorithm>  // std::copy

#include "../collective/communicator-inl.h"
#include "../common/categorical.h"  // common::IsCat
#include "../common/column_matrix.h"
#include "../tree/param.h"        // FIXME(jiamingy): Find a better way to share this parameter.
#include "gradient_index.h"
#include "proxy_dmatrix.h"
#include "simple_batch_iterator.h"

namespace xgboost {
namespace data {
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

  StringView msg{"All batch should be on the same device."};
  if (batch_param_.gpu_id != Context::kCpuId) {
    CHECK_EQ(d, batch_param_.gpu_id) << msg;
  }

  batch_param_ = BatchParam{d, max_bin};
  // hardcoded parameter.
  batch_param_.sparse_thresh = tree::TrainParam::DftSparseThreshold();

  ctx_.UpdateAllowUnknown(
      Args{{"nthread", std::to_string(nthread)}, {"gpu_id", std::to_string(d)}});
  if (ctx_.IsCPU()) {
    this->InitFromCPU(iter_handle, missing, ref);
  } else {
    this->InitFromCUDA(iter_handle, missing, ref);
  }
}

void GetCutsFromRef(std::shared_ptr<DMatrix> ref_, bst_feature_t n_features, BatchParam p,
                    common::HistogramCuts* p_cuts) {
  CHECK(ref_);
  CHECK(p_cuts);
  auto csr = [&]() {
    for (auto const& page : ref_->GetBatches<GHistIndexMatrix>(p)) {
      *p_cuts = page.cut;
      break;
    }
  };
  auto ellpack = [&]() {
    for (auto const& page : ref_->GetBatches<EllpackPage>(p)) {
      GetCutsFromEllpack(page, p_cuts);
      break;
    }
  };

  if (ref_->PageExists<GHistIndexMatrix>()) {
    csr();
  } else if (ref_->PageExists<EllpackPage>()) {
    ellpack();
  } else {
    if (p.gpu_id == Context::kCpuId) {
      csr();
    } else {
      ellpack();
    }
  }
  CHECK_EQ(ref_->Info().num_col_, n_features)
      << "Invalid ref DMatrix, different number of features.";
}

void IterativeDMatrix::InitFromCPU(DataIterHandle iter_handle, float missing,
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

  std::vector<size_t> column_sizes;
  auto const is_valid = data::IsValidFunctor{missing};
  auto nnz_cnt = [&]() {
    return HostAdapterDispatch(proxy, [&](auto const& value) {
      size_t n_threads = ctx_.Threads();
      size_t n_features = column_sizes.size();
      linalg::Tensor<size_t, 2> column_sizes_tloc({n_threads, n_features}, Context::kCpuId);
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

  size_t n_features = 0;
  size_t n_batches = 0;
  size_t accumulated_rows{0};
  size_t nnz{0};

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
      column_sizes.resize(n_features);
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
  collective::Allreduce<collective::Operation::kMax>(&info_.num_col_, 1);
  CHECK(std::none_of(column_sizes.cbegin(), column_sizes.cend(), [&](auto f) {
    return f > accumulated_rows;
  })) << "Something went wrong during iteration.";

  CHECK_GE(n_features, 1) << "Data must has at least 1 column.";

  /**
   * Generate quantiles
   */
  accumulated_rows = 0;
  if (ref) {
    GetCutsFromRef(ref, Info().num_col_, batch_param_, &cuts);
  } else {
    size_t i = 0;
    while (iter.Next()) {
      if (!p_sketch) {
        p_sketch.reset(new common::HostSketchContainer{batch_param_.max_bin,
                                                       proxy->Info().feature_types.ConstHostSpan(),
                                                       column_sizes, false, ctx_.Threads()});
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
    p_sketch->MakeCuts(&cuts);
  }

  /**
   * Generate gradient index.
   */
  this->ghist_ = std::make_unique<GHistIndexMatrix>(Info(), std::move(cuts), batch_param_.max_bin);
  size_t rbegin = 0;
  size_t prev_sum = 0;
  size_t i = 0;
  while (iter.Next()) {
    HostAdapterDispatch(proxy, [&](auto const& batch) {
      proxy->Info().num_nonzero_ = batch_nnz[i];
      this->ghist_->PushAdapterBatch(&ctx_, rbegin, prev_sum, batch, missing,
                                     proxy->Info().feature_types.ConstHostSpan(),
                                     batch_param_.sparse_thresh, Info().num_row_);
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

  /**
   * Generate column matrix
   */
  accumulated_rows = 0;
  while (iter.Next()) {
    HostAdapterDispatch(proxy, [&](auto const& batch) {
      this->ghist_->PushAdapterBatchColumns(&ctx_, batch, missing, accumulated_rows);
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
}

BatchSet<GHistIndexMatrix> IterativeDMatrix::GetGradientIndex(BatchParam const& param) {
  CheckParam(param);
  if (!ghist_) {
    CHECK(ellpack_);
    ghist_ = std::make_shared<GHistIndexMatrix>(&ctx_, Info(), *ellpack_, param);
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

BatchSet<ExtSparsePage> IterativeDMatrix::GetExtBatches(BatchParam const& param) {
  for (auto const& page : this->GetGradientIndex(param)) {
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
}  // namespace data
}  // namespace xgboost
