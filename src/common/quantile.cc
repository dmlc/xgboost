/*!
 * Copyright 2020-2022 by XGBoost Contributors
 */
#include "quantile.h"

#include <limits>
#include <utility>

#include "categorical.h"
#include "hist_util.h"
#include "rabit/rabit.h"

namespace xgboost {
namespace common {

template <typename WQSketch>
SketchContainerImpl<WQSketch>::SketchContainerImpl(std::vector<bst_row_t> columns_size,
                                                   int32_t max_bins,
                                                   Span<FeatureType const> feature_types,
                                                   bool use_group, int32_t n_threads)
    : feature_types_(feature_types.cbegin(), feature_types.cend()),
      columns_size_{std::move(columns_size)},
      max_bins_{max_bins},
      use_group_ind_{use_group},
      n_threads_{n_threads} {
  monitor_.Init(__func__);
  CHECK_NE(columns_size_.size(), 0);
  sketches_.resize(columns_size_.size());
  CHECK_GE(n_threads_, 1);
  categories_.resize(columns_size_.size());
  has_categorical_ = std::any_of(feature_types_.cbegin(), feature_types_.cend(), IsCatOp{});
}

template <typename WQSketch>
std::vector<bst_row_t> SketchContainerImpl<WQSketch>::CalcColumnSize(SparsePage const &batch,
                                                                     bst_feature_t const n_columns,
                                                                     size_t const nthreads) {
  auto page = batch.GetView();
  std::vector<std::vector<bst_row_t>> column_sizes(nthreads);
  for (auto &column : column_sizes) {
    column.resize(n_columns, 0);
  }

  ParallelFor(page.Size(), nthreads, [&](omp_ulong i) {
    auto &local_column_sizes = column_sizes.at(omp_get_thread_num());
    auto row = page[i];
    auto const *p_row = row.data();
    for (size_t j = 0; j < row.size(); ++j) {
      local_column_sizes.at(p_row[j].index)++;
    }
  });
  std::vector<bst_row_t> entries_per_columns(n_columns, 0);
  ParallelFor(n_columns, nthreads, [&](bst_omp_uint i) {
    for (auto const &thread : column_sizes) {
      entries_per_columns[i] += thread[i];
    }
  });
  return entries_per_columns;
}

template <typename WQSketch>
std::vector<bst_feature_t> SketchContainerImpl<WQSketch>::LoadBalance(SparsePage const &batch,
                                                                      bst_feature_t n_columns,
                                                                      size_t const nthreads) {
  /* Some sparse datasets have their mass concentrating on small number of features.  To
   * avoid waiting for a few threads running forever, we here distribute different number
   * of columns to different threads according to number of entries.
   */
  auto page = batch.GetView();
  size_t const total_entries = page.data.size();
  size_t const entries_per_thread = DivRoundUp(total_entries, nthreads);

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
std::vector<float> MergeWeights(MetaInfo const &info, Span<float const> hessian, bool use_group,
                                int32_t n_threads) {
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

template <typename WQSketch>
void SketchContainerImpl<WQSketch>::PushRowPage(SparsePage const &page, MetaInfo const &info,
                                                Span<float const> hessian) {
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
              if (IsCat(feature_types_, ii)) {
                categories_[ii].emplace(p_inst[ii].fvalue);
              } else {
                sketches_[ii].Push(p_inst[ii].fvalue, w);
              }
            }
          } else {
            for (size_t i = 0; i < inst.size(); ++i) {
              auto const& entry = p_inst[i];
              if (entry.index >= begin && entry.index < end) {
                if (IsCat(feature_types_, entry.index)) {
                  categories_[entry.index].emplace(entry.fvalue);
                } else {
                  sketches_[entry.index].Push(entry.fvalue, w);
                }
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

namespace {
/**
 * \brief A view over gathered sketch values.
 */
template <typename T>
struct QuantileAllreduce {
  common::Span<T> global_values;
  common::Span<size_t> worker_indptr;
  common::Span<size_t> feature_indptr;
  size_t n_features{0};
  /**
   * \brief Get sketch values of the a feature from a worker.
   *
   * \param rank rank of target worker
   * \param fidx feature idx
   */
  auto Values(int32_t rank, bst_feature_t fidx) const {
    // get span for worker
    auto wsize = worker_indptr[rank + 1] - worker_indptr[rank];
    auto worker_values = global_values.subspan(worker_indptr[rank], wsize);
    auto psize = n_features + 1;
    auto worker_feat_indptr = feature_indptr.subspan(psize * rank, psize);
    // get span for feature
    auto feat_beg = worker_feat_indptr[fidx];
    auto feat_size = worker_feat_indptr[fidx + 1] - feat_beg;
    return worker_values.subspan(feat_beg, feat_size);
  }
};

/**
 * \brief Merge all categories from other workers.
 */
void AllreduceCategories(Span<FeatureType const> feature_types, int32_t n_threads,
                         std::vector<std::set<float>> *p_categories) {
  auto &categories = *p_categories;
  auto world_size = rabit::GetWorldSize();
  auto rank = rabit::GetRank();
  if (world_size == 1) {
    return;
  }

  // CSC indptr to each feature
  std::vector<size_t> feature_ptr(categories.size() + 1, 0);
  for (size_t i = 0; i < categories.size(); ++i) {
    auto const &feat = categories[i];
    feature_ptr[i + 1] = feat.size();
  }
  std::partial_sum(feature_ptr.begin(), feature_ptr.end(), feature_ptr.begin());
  CHECK_EQ(feature_ptr.front(), 0);

  // gather all feature ptrs from workers
  std::vector<size_t> global_feat_ptrs(feature_ptr.size() * world_size, 0);
  size_t feat_begin = rank * feature_ptr.size();  // pointer to current worker
  std::copy(feature_ptr.begin(), feature_ptr.end(), global_feat_ptrs.begin() + feat_begin);
  rabit::Allreduce<rabit::op::Sum>(global_feat_ptrs.data(), global_feat_ptrs.size());

  // move all categories into a flatten vector to prepare for allreduce
  size_t total = feature_ptr.back();
  std::vector<float> flatten(total, 0);
  auto cursor{flatten.begin()};
  for (auto const &feat : categories) {
    cursor = std::copy(feat.cbegin(), feat.cend(), cursor);
  }

  // indptr for indexing workers
  std::vector<size_t> global_worker_ptr(world_size + 1, 0);
  global_worker_ptr[rank + 1] = total;  // shift 1 to right for constructing the indptr
  rabit::Allreduce<rabit::op::Sum>(global_worker_ptr.data(), global_worker_ptr.size());
  std::partial_sum(global_worker_ptr.cbegin(), global_worker_ptr.cend(), global_worker_ptr.begin());
  // total number of categories in all workers with all features
  auto gtotal = global_worker_ptr.back();

  // categories in all workers with all features.
  std::vector<float> global_categories(gtotal, 0);
  auto rank_begin = global_worker_ptr[rank];
  auto rank_size = global_worker_ptr[rank + 1] - rank_begin;
  CHECK_EQ(rank_size, total);
  std::copy(flatten.cbegin(), flatten.cend(), global_categories.begin() + rank_begin);
  // gather values from all workers.
  rabit::Allreduce<rabit::op::Sum>(global_categories.data(), global_categories.size());
  QuantileAllreduce<float> allreduce_result{global_categories, global_worker_ptr, global_feat_ptrs,
                                            categories.size()};
  ParallelFor(categories.size(), n_threads, [&](auto fidx) {
    if (!IsCat(feature_types, fidx)) {
      return;
    }
    for (int32_t r = 0; r < world_size; ++r) {
      if (r == rank) {
        // continue if it's current worker.
        continue;
      }
      // 1 feature of 1 worker
      auto worker_feature = allreduce_result.Values(r, fidx);
      for (auto c : worker_feature) {
        categories[fidx].emplace(c);
      }
    }
  });
}
}  // anonymous namespace

template <typename WQSketch>
void SketchContainerImpl<WQSketch>::GatherSketchInfo(
    std::vector<typename WQSketch::SummaryContainer> const &reduced,
    std::vector<size_t> *p_worker_segments, std::vector<bst_row_t> *p_sketches_scan,
    std::vector<typename WQSketch::Entry> *p_global_sketches) {
  auto &worker_segments = *p_worker_segments;
  worker_segments.resize(1, 0);
  auto world = rabit::GetWorldSize();
  auto rank = rabit::GetRank();
  auto n_columns = sketches_.size();

  // get the size of each feature.
  std::vector<bst_row_t> sketch_size;
  for (size_t i = 0; i < reduced.size(); ++i) {
    if (IsCat(feature_types_, i)) {
      sketch_size.push_back(0);
    } else {
      sketch_size.push_back(reduced[i].size);
    }
  }
  // turn the size into CSC indptr
  std::vector<bst_row_t> &sketches_scan = *p_sketches_scan;
  sketches_scan.resize((n_columns + 1) * world, 0);
  size_t beg_scan = rank * (n_columns + 1);  // starting storage for current worker.
  std::partial_sum(sketch_size.cbegin(), sketch_size.cend(), sketches_scan.begin() + beg_scan + 1);

  // Gather all column pointers
  rabit::Allreduce<rabit::op::Sum>(sketches_scan.data(), sketches_scan.size());
  for (int32_t i = 0; i < world; ++i) {
    size_t back = (i + 1) * (n_columns + 1) - 1;
    auto n_entries = sketches_scan.at(back);
    worker_segments.push_back(n_entries);
  }
  // Offset of sketch from each worker.
  std::partial_sum(worker_segments.begin(), worker_segments.end(), worker_segments.begin());
  CHECK_GE(worker_segments.size(), 1);
  auto total = worker_segments.back();

  auto &global_sketches = *p_global_sketches;
  global_sketches.resize(total, typename WQSketch::Entry{0, 0, 0, 0});
  auto worker_sketch = Span<typename WQSketch::Entry>{global_sketches}.subspan(
      worker_segments[rank], worker_segments[rank + 1] - worker_segments[rank]);
  auto cursor{worker_sketch.begin()};
  for (size_t fidx = 0; fidx < reduced.size(); ++fidx) {
    auto const &sketch = reduced[fidx];
    if (IsCat(feature_types_, fidx)) {
      // nothing to do if it's categorical feature, size is 0 so no need to change cursor
      continue;
    } else {
      cursor = std::copy(sketch.data, sketch.data + sketch.size, cursor);
    }
  }

  static_assert(sizeof(typename WQSketch::Entry) / 4 == sizeof(float),
                "Unexpected size of sketch entry.");
  rabit::Allreduce<rabit::op::Sum>(
      reinterpret_cast<float *>(global_sketches.data()),
      global_sketches.size() * sizeof(typename WQSketch::Entry) / sizeof(float));
}

template <typename WQSketch>
void SketchContainerImpl<WQSketch>::AllReduce(
    std::vector<typename WQSketch::SummaryContainer> *p_reduced,
    std::vector<int32_t>* p_num_cuts) {
  monitor_.Start(__func__);

  size_t n_columns = sketches_.size();
  rabit::Allreduce<rabit::op::Max>(&n_columns, 1);
  CHECK_EQ(n_columns, sketches_.size()) << "Number of columns differs across workers";

  AllreduceCategories(feature_types_, n_threads_, &categories_);

  auto& num_cuts = *p_num_cuts;
  CHECK_EQ(num_cuts.size(), 0);
  num_cuts.resize(sketches_.size());

  auto &reduced = *p_reduced;
  reduced.resize(sketches_.size());

  // Prune the intermediate num cuts for synchronization.
  std::vector<bst_row_t> global_column_size(columns_size_);
  rabit::Allreduce<rabit::op::Sum>(global_column_size.data(), global_column_size.size());

  ParallelFor(sketches_.size(), n_threads_, [&](size_t i) {
    int32_t intermediate_num_cuts = static_cast<int32_t>(
        std::min(global_column_size[i], static_cast<size_t>(max_bins_ * WQSketch::kFactor)));
    if (global_column_size[i] == 0) {
      return;
    }
    if (IsCat(feature_types_, i)) {
      intermediate_num_cuts = categories_[i].size();
    } else {
      typename WQSketch::SummaryContainer out;
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

  std::vector<typename WQSketch::Entry> global_sketches;
  this->GatherSketchInfo(reduced, &worker_segments, &sketches_scan, &global_sketches);

  std::vector<typename WQSketch::SummaryContainer> final_sketches(n_columns);

  ParallelFor(n_columns, n_threads_, [&](auto fidx) {
    // gcc raises subobject-linkage warning if we put allreduce_result as lambda capture
    QuantileAllreduce<typename WQSketch::Entry> allreduce_result{global_sketches, worker_segments,
                                                                 sketches_scan, n_columns};
    int32_t intermediate_num_cuts = num_cuts[fidx];
    auto nbytes = WQSketch::SummaryContainer::CalcMemCost(intermediate_num_cuts);
    if (IsCat(feature_types_, fidx)) {
      return;
    }

    for (int32_t r = 0; r < world; ++r) {
      // 1 feature of 1 worker
      auto worker_feature = allreduce_result.Values(r, fidx);
      CHECK(worker_feature.data());
      typename WQSketch::Summary summary(worker_feature.data(), worker_feature.size());
      auto &out = final_sketches.at(fidx);
      out.Reduce(summary, nbytes);
    }

    reduced.at(fidx).Reserve(intermediate_num_cuts);
    reduced.at(fidx).SetPrune(final_sketches.at(fidx), intermediate_num_cuts);
  });
  monitor_.Stop(__func__);
}

template <typename SketchType>
void AddCutPoint(typename SketchType::SummaryContainer const &summary, int max_bin,
                 HistogramCuts *cuts) {
  size_t required_cuts = std::min(summary.size, static_cast<size_t>(max_bin));
  auto &cut_values = cuts->cut_values_.HostVector();
  for (size_t i = 1; i < required_cuts; ++i) {
    bst_float cpt = summary.data[i].value;
    if (i == 1 || cpt > cut_values.back()) {
      cut_values.push_back(cpt);
    }
  }
}

auto AddCategories(std::set<float> const &categories, HistogramCuts *cuts) {
  if (std::any_of(categories.cbegin(), categories.cend(), InvalidCat)) {
    InvalidCategory();
  }
  auto &cut_values = cuts->cut_values_.HostVector();
  auto max_cat = *std::max_element(categories.cbegin(), categories.cend());
  CheckMaxCat(max_cat, categories.size());
  for (bst_cat_t i = 0; i <= AsCat(max_cat); ++i) {
    cut_values.push_back(i);
  }
  return max_cat;
}

template <typename WQSketch>
void SketchContainerImpl<WQSketch>::MakeCuts(HistogramCuts* cuts) {
  monitor_.Start(__func__);
  std::vector<typename WQSketch::SummaryContainer> reduced;
  std::vector<int32_t> num_cuts;
  this->AllReduce(&reduced, &num_cuts);

  cuts->min_vals_.HostVector().resize(sketches_.size(), 0.0f);
  std::vector<typename WQSketch::SummaryContainer> final_summaries(reduced.size());

  ParallelFor(reduced.size(), n_threads_, Sched::Guided(), [&](size_t fidx) {
    if (IsCat(feature_types_, fidx)) {
      return;
    }
    typename WQSketch::SummaryContainer &a = final_summaries[fidx];
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

  float max_cat{-1.f};
  for (size_t fid = 0; fid < reduced.size(); ++fid) {
    size_t max_num_bins = std::min(num_cuts[fid], max_bins_);
    typename WQSketch::SummaryContainer const& a = final_summaries[fid];
    if (IsCat(feature_types_, fid)) {
      max_cat = std::max(max_cat, AddCategories(categories_.at(fid), cuts));
    } else {
      AddCutPoint<WQSketch>(a, max_num_bins, cuts);
      // push a value that is greater than anything
      const bst_float cpt = (a.size > 0) ? a.data[a.size - 1].value
                                         : cuts->min_vals_.HostVector()[fid];
      // this must be bigger than last value in a scale
      const bst_float last = cpt + (fabs(cpt) + 1e-5f);
      cuts->cut_values_.HostVector().push_back(last);
    }

    // Ensure that every feature gets at least one quantile point
    CHECK_LE(cuts->cut_values_.HostVector().size(), std::numeric_limits<uint32_t>::max());
    auto cut_size = static_cast<uint32_t>(cuts->cut_values_.HostVector().size());
    CHECK_GT(cut_size, cuts->cut_ptrs_.HostVector().back());
    cuts->cut_ptrs_.HostVector().push_back(cut_size);
  }

  cuts->SetCategorical(this->has_categorical_, max_cat);
  monitor_.Stop(__func__);
}

template class SketchContainerImpl<WQuantileSketch<float, float>>;
template class SketchContainerImpl<WXQuantileSketch<float, float>>;

HostSketchContainer::HostSketchContainer(int32_t max_bins, MetaInfo const &info,
                                         std::vector<size_t> columns_size, bool use_group,
                                         Span<float const> hessian, int32_t n_threads)
    : SketchContainerImpl{columns_size, max_bins, info.feature_types.ConstHostSpan(), use_group,
                          n_threads} {
  monitor_.Init(__func__);
  ParallelFor(sketches_.size(), n_threads_, Sched::Auto(), [&](auto i) {
    auto n_bins = std::min(static_cast<size_t>(max_bins_), columns_size_[i]);
    n_bins = std::max(n_bins, static_cast<decltype(n_bins)>(1));
    auto eps = 1.0 / (static_cast<float>(n_bins) * WQSketch::kFactor);
    if (!IsCat(this->feature_types_, i)) {
      sketches_[i].Init(columns_size_[i], eps);
      sketches_[i].inqueue.queue.resize(sketches_[i].limit_size * 2);
    }
  });
}

void SortedSketchContainer::PushColPage(SparsePage const &page, MetaInfo const &info,
                                        Span<float const> hessian) {
  monitor_.Start(__func__);
  // glue these conditions using ternary operator to avoid making data copies.
  auto const &weights =
      hessian.empty() ? (use_group_ind_ ? UnrollGroupWeights(info)     // use group weight
                                        : info.weights_.HostVector())  // use sample weight
                      : MergeWeights(info, hessian, use_group_ind_,
                                     n_threads_);  // use hessian merged with group/sample weights
  CHECK_EQ(weights.size(), info.num_row_);

  auto view = page.GetView();
  ParallelFor(view.Size(), n_threads_, [&](size_t fidx) {
    auto column = view[fidx];
    auto &sketch = sketches_[fidx];
    sketch.Init(max_bins_);
    // first pass
    sketch.sum_total = 0.0;
    for (auto c : column) {
      sketch.sum_total += weights[c.index];
    }
    // second pass
    if (IsCat(feature_types_, fidx)) {
      for (auto c : column) {
        categories_[fidx].emplace(c.fvalue);
      }
    } else {
      for (auto c : column) {
        sketch.Push(c.fvalue, weights[c.index], max_bins_);
      }
    }

    if (!IsCat(feature_types_, fidx) && !column.empty()) {
      sketch.Finalize(max_bins_);
    }
  });
  monitor_.Stop(__func__);
}
}  // namespace common
}  // namespace xgboost
