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
                                         int32_t max_bins, bool use_group,
                                         int32_t n_threads)
    : columns_size_{std::move(columns_size)}, max_bins_{max_bins},
      use_group_ind_{use_group}, n_threads_{n_threads} {
  monitor_.Init(__func__);
  CHECK_NE(columns_size_.size(), 0);
  sketches_.resize(columns_size_.size());
  CHECK_GE(n_threads_, 1);
  ParallelFor(sketches_.size(), n_threads_, Sched::Auto(), [&](auto i) {
    auto n_bins = std::min(static_cast<size_t>(max_bins_), columns_size_[i]);
    n_bins = std::max(n_bins, static_cast<decltype(n_bins)>(1));
    auto eps = 1.0 / (static_cast<float>(n_bins) * WQSketch::kFactor);
    sketches_[i].Init(columns_size_[i], eps);
    sketches_[i].inqueue.queue.resize(sketches_[i].limit_size * 2);
  });
}

std::vector<bst_row_t>
HostSketchContainer::CalcColumnSize(SparsePage const &batch,
                                    bst_feature_t const n_columns,
                                    size_t const nthreads) {
  auto page = batch.GetView();
  std::vector<std::vector<bst_row_t>> column_sizes(nthreads);
  for (auto &column : column_sizes) {
    column.resize(n_columns, 0);
  }

  ParallelFor(omp_ulong(page.Size()), nthreads, [&](omp_ulong i) {
    auto &local_column_sizes = column_sizes.at(omp_get_thread_num());
    auto row = page[i];
    auto const *p_row = row.data();
    for (size_t j = 0; j < row.size(); ++j) {
      local_column_sizes.at(p_row[j].index)++;
    }
  });
  std::vector<bst_row_t> entries_per_columns(n_columns, 0);
  ParallelFor(bst_omp_uint(n_columns), nthreads, [&](bst_omp_uint i) {
    for (auto const &thread : column_sizes) {
      entries_per_columns[i] += thread[i];
    }
  });
  return entries_per_columns;
}

std::vector<bst_feature_t> HostSketchContainer::LoadBalance(
    SparsePage const &batch, bst_feature_t n_columns, size_t const nthreads) {
  /* Some sparse datasets have their mass concentrating on small number of features.  To
   * avoid waiting for a few threads running forever, we here distribute different number
   * of columns to different threads according to number of entries.
   */
  auto page = batch.GetView();
  size_t const total_entries = page.data.size();
  size_t const entries_per_thread = common::DivRoundUp(total_entries, nthreads);

  std::vector<std::vector<bst_row_t>> column_sizes(nthreads);
  for (auto& column : column_sizes) {
    column.resize(n_columns, 0);
  }
  std::vector<bst_row_t> entries_per_columns =
      CalcColumnSize(batch, n_columns, nthreads);
  std::vector<bst_feature_t> cols_ptr(nthreads + 1, 0);
  size_t count {0};
  size_t current_thread {1};

  for (auto col : entries_per_columns) {
    cols_ptr.at(current_thread)++;  // add one column to thread
    count += col;
    CHECK_LE(count, total_entries);
    if (count > entries_per_thread) {
      current_thread++;
      count = 0;
      cols_ptr.at(current_thread) = cols_ptr[current_thread-1];
    }
  }
  // Idle threads.
  for (; current_thread < cols_ptr.size() - 1; ++current_thread) {
    cols_ptr[current_thread+1] = cols_ptr[current_thread];
  }
  return cols_ptr;
}

namespace {
// Function to merge hessian and sample weights
std::vector<float> MergeWeights(MetaInfo const &info,
                                Span<float> const hessian,
                                bool use_group, int32_t n_threads) {
  CHECK_EQ(hessian.size(), info.num_row_);
  std::vector<float> results(hessian.size());
  auto const &group_ptr = info.group_ptr_;
  auto const& weights = info.weights_.HostVector();
  auto get_weight = [&](size_t i) { return weights.empty() ? 1.0f : weights[i]; };
  if (use_group) {
    CHECK_GE(group_ptr.size(), 2);
    CHECK_EQ(group_ptr.back(), hessian.size());
    size_t cur_group = 0;
    for (size_t i = 0; i < hessian.size(); ++i) {
      results[i] = hessian[i] * get_weight(cur_group);
      if (i == group_ptr[cur_group + 1]) {
        cur_group++;
      }
    }
  } else {
    ParallelFor(hessian.size(), n_threads, Sched::Auto(),
                [&](auto i) { results[i] = hessian[i] * get_weight(i); });
  }
  return results;
}

std::vector<float> UnrollGroupWeights(MetaInfo const &info) {
  std::vector<float> const &group_weights = info.weights_.HostVector();
  if (group_weights.empty()) {
    return group_weights;
  }

  size_t n_samples = info.num_row_;
  auto const &group_ptr = info.group_ptr_;
  std::vector<float> results(n_samples);
  CHECK_GE(group_ptr.size(), 2);
  CHECK_EQ(group_ptr.back(), n_samples);
  size_t cur_group = 0;
  for (size_t i = 0; i < n_samples; ++i) {
    results[i] = group_weights[cur_group];
    if (i == group_ptr[cur_group + 1]) {
      cur_group++;
    }
  }
  return results;
}
}  // anonymous namespace

void HostSketchContainer::PushRowPage(
    SparsePage const &page, MetaInfo const &info, Span<float> hessian) {
  monitor_.Start(__func__);
  bst_feature_t n_columns = info.num_col_;
  auto is_dense = info.num_nonzero_ == info.num_col_ * info.num_row_;
  CHECK_GE(n_threads_, 1);
  CHECK_EQ(sketches_.size(), n_columns);

  // glue these conditions using ternary operator to avoid making data copies.
  auto const &weights =
      hessian.empty()
          ? (use_group_ind_ ? UnrollGroupWeights(info)     // use group weight
                            : info.weights_.HostVector())  // use sample weight
          : MergeWeights(
                info, hessian, use_group_ind_,
                n_threads_);  // use hessian merged with group/sample weights
  if (!weights.empty()) {
    CHECK_EQ(weights.size(), info.num_row_);
  }

  auto batch = page.GetView();
  // Parallel over columns.  Each thread owns a set of consecutive columns.
  auto const ncol = static_cast<bst_feature_t>(info.num_col_);
  auto thread_columns_ptr = LoadBalance(page, info.num_col_, n_threads_);

  dmlc::OMPException exc;
#pragma omp parallel num_threads(n_threads_)
  {
    exc.Run([&]() {
      auto tid = static_cast<uint32_t>(omp_get_thread_num());
      auto const begin = thread_columns_ptr[tid];
      auto const end = thread_columns_ptr[tid + 1];

      // do not iterate if no columns are assigned to the thread
      if (begin < end && end <= ncol) {
        for (size_t i = 0; i < batch.Size(); ++i) {
          size_t const ridx = page.base_rowid + i;
          SparsePage::Inst const inst = batch[i];
          auto w = weights.empty() ? 1.0f : weights[ridx];
          auto p_inst = inst.data();
          if (is_dense) {
            for (size_t ii = begin; ii < end; ii++) {
              sketches_[ii].Push(p_inst[ii].fvalue, w);
            }
          } else {
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
  exc.Rethrow();
  monitor_.Stop(__func__);
}

void HostSketchContainer::GatherSketchInfo(
    std::vector<WQSketch::SummaryContainer> const &reduced,
    std::vector<size_t> *p_worker_segments,
    std::vector<bst_row_t> *p_sketches_scan,
    std::vector<WQSketch::Entry> *p_global_sketches) {
  auto& worker_segments = *p_worker_segments;
  worker_segments.resize(1, 0);
  auto world = rabit::GetWorldSize();
  auto rank = rabit::GetRank();
  auto n_columns = sketches_.size();

  std::vector<bst_row_t> sketch_size;
  for (auto const& sketch : reduced) {
    sketch_size.push_back(sketch.size);
  }
  std::vector<bst_row_t>& sketches_scan = *p_sketches_scan;
  sketches_scan.resize((n_columns + 1) * world, 0);
  size_t beg_scan = rank * (n_columns + 1);
  std::partial_sum(sketch_size.cbegin(), sketch_size.cend(),
                   sketches_scan.begin() + beg_scan + 1);
  // Gather all column pointers
  rabit::Allreduce<rabit::op::Sum>(sketches_scan.data(), sketches_scan.size());

  for (int32_t i = 0; i < world; ++i) {
    size_t back = (i + 1) * (n_columns + 1) - 1;
    auto n_entries = sketches_scan.at(back);
    worker_segments.push_back(n_entries);
  }
  // Offset of sketch from each worker.
  std::partial_sum(worker_segments.begin(), worker_segments.end(),
                   worker_segments.begin());
  CHECK_GE(worker_segments.size(), 1);
  auto total = worker_segments.back();

  auto& global_sketches = *p_global_sketches;
  global_sketches.resize(total, WQSketch::Entry{0, 0, 0, 0});
  auto worker_sketch = Span<WQSketch::Entry>{global_sketches}.subspan(
      worker_segments[rank], worker_segments[rank + 1] - worker_segments[rank]);
  size_t cursor = 0;
  for (auto const &sketch : reduced) {
    std::copy(sketch.data, sketch.data + sketch.size,
              worker_sketch.begin() + cursor);
    cursor += sketch.size;
  }

  static_assert(sizeof(WQSketch::Entry) / 4 == sizeof(float), "");
  rabit::Allreduce<rabit::op::Sum>(
      reinterpret_cast<float *>(global_sketches.data()),
      global_sketches.size() * sizeof(WQSketch::Entry) / sizeof(float));
}

void HostSketchContainer::AllReduce(
    std::vector<WQSketch::SummaryContainer> *p_reduced,
    std::vector<int32_t>* p_num_cuts) {
  monitor_.Start(__func__);
  auto& num_cuts = *p_num_cuts;
  CHECK_EQ(num_cuts.size(), 0);
  num_cuts.resize(sketches_.size());

  auto &reduced = *p_reduced;
  reduced.resize(sketches_.size());

  size_t n_columns = sketches_.size();
  rabit::Allreduce<rabit::op::Max>(&n_columns, 1);
  CHECK_EQ(n_columns, sketches_.size()) << "Number of columns differs across workers";

  // Prune the intermediate num cuts for synchronization.
  std::vector<bst_row_t> global_column_size(columns_size_);
  rabit::Allreduce<rabit::op::Sum>(global_column_size.data(), global_column_size.size());

  ParallelFor(sketches_.size(), n_threads_, [&](size_t i) {
    int32_t intermediate_num_cuts = static_cast<int32_t>(
        std::min(global_column_size[i],
                 static_cast<size_t>(max_bins_ * WQSketch::kFactor)));
    if (global_column_size[i] != 0) {
      WQSketch::SummaryContainer out;
      sketches_[i].GetSummary(&out);
      reduced[i].Reserve(intermediate_num_cuts);
      CHECK(reduced[i].data);
      reduced[i].SetPrune(out, intermediate_num_cuts);
    }
    num_cuts[i] = intermediate_num_cuts;
  });

  auto world = rabit::GetWorldSize();
  if (world == 1) {
    monitor_.Stop(__func__);
    return;
  }

  std::vector<size_t> worker_segments(1, 0);  // CSC pointer to sketches.
  std::vector<bst_row_t> sketches_scan((n_columns + 1) * world, 0);

  std::vector<WQSketch::Entry> global_sketches;
  this->GatherSketchInfo(reduced, &worker_segments, &sketches_scan,
                         &global_sketches);

  std::vector<WQSketch::SummaryContainer> final_sketches(n_columns);
  ParallelFor(n_columns, n_threads_, [&](auto fidx) {
    int32_t intermediate_num_cuts = num_cuts[fidx];
    auto nbytes =
        WQSketch::SummaryContainer::CalcMemCost(intermediate_num_cuts);

    for (int32_t i = 1; i < world + 1; ++i) {
      auto size = worker_segments.at(i) - worker_segments[i - 1];
      auto worker_sketches = Span<WQSketch::Entry>{global_sketches}.subspan(
          worker_segments[i - 1], size);
      auto worker_scan =
          Span<bst_row_t>(sketches_scan)
              .subspan((i - 1) * (n_columns + 1), (n_columns + 1));

      auto worker_feature = worker_sketches.subspan(
          worker_scan[fidx], worker_scan[fidx + 1] - worker_scan[fidx]);
      CHECK(worker_feature.data());
      WQSummary<float, float> summary(worker_feature.data(),
                                      worker_feature.size());
      auto &out = final_sketches.at(fidx);
      out.Reduce(summary, nbytes);
    }

    reduced.at(fidx).Reserve(intermediate_num_cuts);
    reduced.at(fidx).SetPrune(final_sketches.at(fidx), intermediate_num_cuts);
  });
  monitor_.Stop(__func__);
}

void AddCutPoint(WQuantileSketch<float, float>::SummaryContainer const &summary,
                 int max_bin, HistogramCuts *cuts) {
  size_t required_cuts = std::min(summary.size, static_cast<size_t>(max_bin));
  auto& cut_values = cuts->cut_values_.HostVector();
  for (size_t i = 1; i < required_cuts; ++i) {
    bst_float cpt = summary.data[i].value;
    if (i == 1 || cpt > cut_values.back()) {
      cut_values.push_back(cpt);
    }
  }
}

void HostSketchContainer::MakeCuts(HistogramCuts* cuts) {
  monitor_.Start(__func__);
  std::vector<WQSketch::SummaryContainer> reduced;
  std::vector<int32_t> num_cuts;
  this->AllReduce(&reduced, &num_cuts);

  cuts->min_vals_.HostVector().resize(sketches_.size(), 0.0f);
  std::vector<WQSketch::SummaryContainer> final_summaries(reduced.size());

  ParallelFor(reduced.size(), n_threads_, Sched::Guided(), [&](size_t fidx) {
    WQSketch::SummaryContainer &a = final_summaries[fidx];
    size_t max_num_bins = std::min(num_cuts[fidx], max_bins_);
    a.Reserve(max_num_bins + 1);
    CHECK(a.data);
    if (num_cuts[fidx] != 0) {
      a.SetPrune(reduced[fidx], max_num_bins + 1);
      CHECK(a.data && reduced[fidx].data);
      const bst_float mval = a.data[0].value;
      cuts->min_vals_.HostVector()[fidx] = mval - fabs(mval) - 1e-5f;
    } else {
      // Empty column.
      const float mval = 1e-5f;
      cuts->min_vals_.HostVector()[fidx] = mval;
    }
  });

  for (size_t fid = 0; fid < reduced.size(); ++fid) {
    size_t max_num_bins = std::min(num_cuts[fid], max_bins_);
    WQSketch::SummaryContainer const& a = final_summaries[fid];
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
  monitor_.Stop(__func__);
}
}  // namespace common
}  // namespace xgboost
