/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include <limits>
#include <utility>
#include "quantile.h"
#include "hist_util.h"

namespace xgboost {
namespace common {

HostSketchContainer::HostSketchContainer(std::vector<bst_row_t> columns_size,
                                         int32_t max_bins, bool use_group)
    : columns_size_{std::move(columns_size)}, max_bins_{max_bins},
      use_group_ind_{use_group} {
  CHECK_NE(columns_size_.size(), 0);
  sketches_.resize(columns_size_.size());
  for (size_t i = 0; i < sketches_.size(); ++i) {
    auto n_bins = std::min(static_cast<size_t>(max_bins_), columns_size_[i]);
    n_bins = std::max(n_bins, static_cast<decltype(n_bins)>(1));
    auto eps = 1.0 / (static_cast<float>(n_bins) * WQSketch::kFactor);
    sketches_[i].Init(columns_size_[i], eps);
    sketches_[i].inqueue.queue.resize(sketches_[i].limit_size * 2);
  }
}

std::vector<bst_feature_t> LoadBalance(SparsePage const &page,
                                       std::vector<size_t> columns_size,
                                       size_t const nthreads) {
  /* Some sparse datasets have their mass concentrating on small
   * number of features.  To avoid wating for a few threads running
   * forever, we here distirbute different number of columns to
   * different threads according to number of entries. */
  size_t const total_entries = page.data.Size();
  size_t const entries_per_thread = common::DivRoundUp(total_entries, nthreads);

  std::vector<bst_feature_t> cols_ptr(nthreads+1, 0);
  size_t count {0};
  size_t current_thread {1};

  for (auto col : columns_size) {
    cols_ptr[current_thread]++;  // add one column to thread
    count += col;
    if (count > entries_per_thread + 1) {
      current_thread++;
      count = 0;
      cols_ptr[current_thread] = cols_ptr[current_thread-1];
    }
  }
  // Idle threads.
  for (; current_thread < cols_ptr.size() - 1; ++current_thread) {
    cols_ptr[current_thread+1] = cols_ptr[current_thread];
  }

  return cols_ptr;
}

void HostSketchContainer::PushRowPage(SparsePage const &page,
                                      MetaInfo const &info) {
  int nthread = omp_get_max_threads();
  CHECK_EQ(sketches_.size(), info.num_col_);

  // Data groups, used in ranking.
  std::vector<bst_uint> const &group_ptr = info.group_ptr_;
  // Use group index for weights?
  auto batch = page.GetView();
  dmlc::OMPException exec;
  // Parallel over columns.  Asumming the data is dense, each thread owns a set of
  // consecutive columns.
  auto const ncol = static_cast<uint32_t>(info.num_col_);
  auto const is_dense = info.num_nonzero_ == info.num_col_ * info.num_row_;
  auto thread_columns_ptr = LoadBalance(page, columns_size_, nthread);

#pragma omp parallel num_threads(nthread)
  {
    exec.Run([&]() {
      auto tid = static_cast<uint32_t>(omp_get_thread_num());
      auto const begin = thread_columns_ptr[tid];
      auto const end = thread_columns_ptr[tid + 1];
      size_t group_ind = 0;

      // do not iterate if no columns are assigned to the thread
      if (begin < end && end <= ncol) {
        for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
          size_t const ridx = page.base_rowid + i;
          SparsePage::Inst const inst = batch[i];
          if (use_group_ind_) {
            group_ind = this->SearchGroupIndFromRow(group_ptr, i + page.base_rowid);
          }
          size_t w_idx = use_group_ind_ ? group_ind : ridx;
          auto w = info.GetWeight(w_idx);
          auto p_data = inst.data();
          if (is_dense) {
            for (size_t ii = begin; ii < end; ii++) {
              sketches_[ii].Push(p_data[ii].fvalue, w);
            }
          } else {
            auto p_inst = inst.data();
            for (size_t i = 0; i < inst.size(); ++i) {
              auto const& entry = p_inst[i];
              if (entry.index >= begin && entry.index < end) {
                sketches_[entry.index].Push(entry.fvalue, w);
              }
            }
          }
        }
      }
    });
  }
  exec.Rethrow();
}

void AddCutPoint(WQuantileSketch<float, float>::SummaryContainer const &summary,
                 int max_bin, HistogramCuts *cuts) {
  size_t required_cuts = std::min(summary.size, static_cast<size_t>(max_bin));
  auto& cut_values = cuts->cut_values_.HostVector();
  for (size_t i = 1; i < required_cuts; ++i) {
    bst_float cpt = summary.data[i].value;
    if (i == 1 || cpt > cuts->cut_values_.ConstHostVector().back()) {
      cut_values.push_back(cpt);
    }
  }
}

void HostSketchContainer::MakeCuts(HistogramCuts* cuts) {
  rabit::Allreduce<rabit::op::Sum>(columns_size_.data(), columns_size_.size());
  std::vector<WQSketch::SummaryContainer> reduced(sketches_.size());
  std::vector<int32_t> num_cuts;
  size_t nbytes = 0;
  for (size_t i = 0; i < sketches_.size(); ++i) {
    int32_t intermediate_num_cuts =  static_cast<int32_t>(std::min(
        columns_size_[i], static_cast<size_t>(max_bins_ * WQSketch::kFactor)));
    if (columns_size_[i] != 0) {
      WQSketch::SummaryContainer out;
      sketches_[i].GetSummary(&out);
      reduced[i].Reserve(intermediate_num_cuts);
      CHECK(reduced[i].data);
      reduced[i].SetPrune(out, intermediate_num_cuts);
    }
    num_cuts.push_back(intermediate_num_cuts);
    nbytes = std::max(
        WQSketch::SummaryContainer::CalcMemCost(intermediate_num_cuts), nbytes);
  }

  if (rabit::IsDistributed()) {
    // FIXME(trivialfis): This call will allocate nbytes * num_columns on rabit, which
    // may generate oom error when data is sparse.  To fix it, we need to:
    //   - gather the column offsets over all workers.
    //   - run rabit::allgather on sketch data to collect all data.
    //   - merge all gathered sketches based on worker offsets and column offsets of data
    //     from each worker.
    // See GPU implementation for details.
    rabit::SerializeReducer<WQSketch::SummaryContainer> sreducer;
    sreducer.Allreduce(dmlc::BeginPtr(reduced), nbytes, reduced.size());
  }

  cuts->min_vals_.HostVector().resize(sketches_.size(), 0.0f);
  for (size_t fid = 0; fid < reduced.size(); ++fid) {
    WQSketch::SummaryContainer a;
    size_t max_num_bins = std::min(num_cuts[fid], max_bins_);
    a.Reserve(max_num_bins + 1);
    CHECK(a.data);
    if (columns_size_[fid] != 0) {
      a.SetPrune(reduced[fid], max_num_bins + 1);
      CHECK(a.data && reduced[fid].data);
      const bst_float mval = a.data[0].value;
      cuts->min_vals_.HostVector()[fid] = mval - fabs(mval) - 1e-5f;
    } else {
      // Empty column.
      const float mval = 1e-5f;
      cuts->min_vals_.HostVector()[fid] = mval;
    }
    AddCutPoint(a, max_num_bins, cuts);
    // push a value that is greater than anything
    const bst_float cpt
      = (a.size > 0) ? a.data[a.size - 1].value : cuts->min_vals_.HostVector()[fid];
    // this must be bigger than last value in a scale
    const bst_float last = cpt + (fabs(cpt) + 1e-5f);
    cuts->cut_values_.HostVector().push_back(last);

    // Ensure that every feature gets at least one quantile point
    CHECK_LE(cuts->cut_values_.HostVector().size(), std::numeric_limits<uint32_t>::max());
    auto cut_size = static_cast<uint32_t>(cuts->cut_values_.HostVector().size());
    CHECK_GT(cut_size, cuts->cut_ptrs_.HostVector().back());
    cuts->cut_ptrs_.HostVector().push_back(cut_size);
  }
}
}  // namespace common
}  // namespace xgboost
