/**
 * Copyright 2020-2025, XGBoost Contributors
 */
#include "quantile.h"

#include <cstddef>  // for byte
#include <cstdint>  // for uint64_t
#include <limits>
#include <numeric>  // for partial_sum
#include <utility>

#include "../collective/aggregator.h"
#include "../common/error_msg.h"  // for InvalidMaxBin
#include "../data/adapter.h"
#include "categorical.h"
#include "hist_util.h"

namespace xgboost::common {
SketchContainerImpl::SketchContainerImpl(Context const *ctx, std::vector<bst_idx_t> columns_size,
                                         bst_bin_t max_bin, Span<FeatureType const> feature_types,
                                         bool use_group)
    : feature_types_(feature_types.cbegin(), feature_types.cend()),
      columns_size_{std::move(columns_size)},
      max_bins_{max_bin},
      use_group_ind_{use_group},
      n_threads_{ctx->Threads()} {
  monitor_.Init(__func__);
  CHECK_GE(max_bin, 2) << error::InvalidMaxBin();
  CHECK_NE(columns_size_.size(), 0);
  sketches_.resize(columns_size_.size());
  CHECK_GE(n_threads_, 1);
  categories_.resize(columns_size_.size());
  has_categorical_ = std::any_of(feature_types_.cbegin(), feature_types_.cend(), IsCatOp{});
}

namespace {
// Function to merge hessian and sample weights
std::vector<float> MergeWeights(MetaInfo const &info, Span<float const> hessian, bool use_group,
                                int32_t n_threads) {
  CHECK_EQ(hessian.size(), info.num_row_);
  std::vector<float> results(hessian.size());
  auto const &group_ptr = info.group_ptr_;
  auto const &weights = info.weights_.HostVector();
  auto get_weight = [&](size_t i) {
    return weights.empty() ? 1.0f : weights[i];
  };
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

struct SketchReducePayload {
  using Entry = WQuantileSketch::Entry;
  using Summary = WQuantileSketch::Summary;
  using SummaryContainer = WQuantileSketch::SummaryContainer;

  [[nodiscard]] static std::vector<std::byte> SerializeFromSummaries(
      Span<bst_feature_t const> numeric_features,
      std::vector<WQuantileSketch::SummaryContainer> const &reduced) {
    std::size_t total_entries = 0;
    for (auto fidx : numeric_features) {
      total_entries += reduced.at(fidx).Size();
    }

    std::vector<std::byte> bytes;
    InitHeader(&bytes, numeric_features.size(), total_entries);

    for (std::size_t i = 0; i < numeric_features.size(); ++i) {
      auto fidx = numeric_features[i];
      auto out_entries = reduced.at(fidx).Entries();
      AppendEntries(&bytes, i, out_entries);
    }
    auto header_bytes = HeaderBytes(numeric_features.size());
    CHECK_EQ((bytes.size() - header_bytes) / sizeof(Entry), total_entries);
    return bytes;
  }

  [[nodiscard]] static std::size_t HeaderBytes(std::size_t n_features) {
    return sizeof(std::uint64_t) + n_features * sizeof(std::uint64_t);
  }

  static void AppendEntries(std::vector<std::byte> *bytes, std::size_t i,
                            Span<Entry const> entries) {
    CHECK(bytes);
    auto count_offset = sizeof(std::uint64_t) + i * sizeof(std::uint64_t);
    CHECK_LE(count_offset + sizeof(std::uint64_t), bytes->size());
    WritePODAt<std::uint64_t>(bytes, count_offset, static_cast<std::uint64_t>(entries.size()));
    if (entries.empty()) {
      return;
    }
    auto entries_bytes = entries.size() * sizeof(Entry);
    auto const *src = reinterpret_cast<std::byte const *>(entries.data());
    bytes->insert(bytes->end(), src, src + entries_bytes);
  }

  static void InitHeader(std::vector<std::byte> *bytes, std::size_t n_features,
                         std::size_t max_entries) {
    CHECK(bytes);
    auto const header_bytes = HeaderBytes(n_features);
    bytes->clear();
    bytes->reserve(header_bytes + max_entries * sizeof(Entry));
    bytes->resize(header_bytes);
    WritePODAt<std::uint64_t>(bytes, 0, static_cast<std::uint64_t>(n_features));
  }

  [[nodiscard]] static SketchReducePayload Parse(Span<std::byte const> bytes) {
    std::size_t cursor = 0;
    auto n_features = ReadPOD<std::uint64_t>(bytes, &cursor);

    std::vector<std::size_t> offsets(n_features + 1, 0);
    for (std::size_t i = 0; i < n_features; ++i) {
      auto n_i = static_cast<std::size_t>(ReadPOD<std::uint64_t>(bytes, &cursor));
      offsets[i + 1] = offsets[i] + n_i;
    }

    auto n_entries = offsets.back();
    auto payload_bytes = n_entries * sizeof(Entry);
    CHECK_EQ(cursor + payload_bytes, bytes.size());

    Entry *entries = nullptr;
    if (n_entries != 0) {
      auto ptr = bytes.data() + cursor;
      auto addr = reinterpret_cast<std::uintptr_t>(ptr);
      CHECK_EQ(addr % alignof(Entry), 0);
      entries = reinterpret_cast<Entry *>(const_cast<std::byte *>(ptr));
    }

    return {std::move(offsets), Span<Entry>{entries, n_entries}};
  }

  [[nodiscard]] std::size_t NumFeatures() const { return offsets_.size() - 1; }
  [[nodiscard]] std::size_t TotalEntries() const { return entries_.size(); }

  [[nodiscard]] Span<Entry> Entries(std::size_t idx) const {
    auto beg = offsets_.at(idx);
    auto end = offsets_.at(idx + 1);
    auto n = end - beg;
    if (n == 0) {
      return Span<Entry>{};
    }
    return {entries_.data() + beg, n};
  }

  [[nodiscard]] Summary SummaryAt(std::size_t idx) const {
    auto entries = this->Entries(idx);
    return {entries, entries.size()};
  }

 private:
  template <typename T>
  static void WritePODAt(std::vector<std::byte> *out, std::size_t offset, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    CHECK(out);
    CHECK_LE(offset + sizeof(T), out->size());
    auto const *src = reinterpret_cast<std::byte const *>(&value);
    std::copy_n(src, sizeof(T), out->begin() + static_cast<std::ptrdiff_t>(offset));
  }

  template <typename T>
  [[nodiscard]] static T ReadPOD(Span<std::byte const> bytes, std::size_t *cursor) {
    static_assert(std::is_trivially_copyable_v<T>);
    CHECK(cursor);
    CHECK_LE(*cursor + sizeof(T), bytes.size());
    T value{};
    auto *dst = reinterpret_cast<std::byte *>(&value);
    std::copy_n(bytes.data() + *cursor, sizeof(T), dst);
    *cursor += sizeof(T);
    return value;
  }

  SketchReducePayload(std::vector<std::size_t> offsets, Span<Entry> entries)
      : offsets_{std::move(offsets)}, entries_{entries} {}

  std::vector<std::size_t> offsets_;
  Span<Entry> entries_;
};
}  // anonymous namespace

void SketchContainerImpl::PushRowPage(SparsePage const &page, MetaInfo const &info,
                                      Span<float const> hessian) {
  monitor_.Start(__func__);
  bst_feature_t n_columns = info.num_col_;
  auto is_dense = info.num_nonzero_ == info.num_col_ * info.num_row_;
  CHECK_GE(n_threads_, 1);
  CHECK_EQ(sketches_.size(), n_columns);

  // glue these conditions using ternary operator to avoid making data copies.
  auto const &weights =
      hessian.empty() ? (use_group_ind_ ? detail::UnrollGroupWeights(info)  // use group weight
                                        : info.weights_.HostVector())       // use sample weight
                      : MergeWeights(info, hessian, use_group_ind_,
                                     n_threads_);  // use hessian merged with group/sample weights
  if (!weights.empty()) {
    CHECK_EQ(weights.size(), info.num_row_);
  }

  auto batch = data::SparsePageAdapterBatch{page.GetView()};
  this->PushRowPageImpl(batch, page.base_rowid, OptionalWeights{weights}, page.data.Size(),
                        info.num_col_, is_dense, [](auto) { return true; });
  monitor_.Stop(__func__);
}

template <typename Batch>
void HostSketchContainer::PushAdapterBatch(Batch const &batch, size_t base_rowid,
                                           MetaInfo const &info, float missing) {
  auto const &h_weights =
      (use_group_ind_ ? detail::UnrollGroupWeights(info) : info.weights_.HostVector());
  if (!use_group_ind_ && !h_weights.empty()) {
    CHECK_EQ(h_weights.size(), batch.Size()) << "Invalid size of sample weight.";
  }

  auto is_valid = data::IsValidFunctor{missing};
  auto weights = OptionalWeights{Span<float const>{h_weights}};
  // the nnz from info is not reliable as sketching might be the first place to go through
  // the data.
  auto is_dense = info.num_nonzero_ == info.num_col_ * info.num_row_;
  CHECK(!this->columns_size_.empty());
  this->PushRowPageImpl(batch, base_rowid, weights, info.num_nonzero_, info.num_col_, is_dense,
                        is_valid);
}

#define INSTANTIATE(_type)                                          \
  template void HostSketchContainer::PushAdapterBatch<data::_type>( \
      data::_type const &batch, size_t base_rowid, MetaInfo const &info, float missing);

INSTANTIATE(ArrayAdapterBatch)
INSTANTIATE(DenseAdapterBatch)
INSTANTIATE(CSRArrayAdapterBatch)
INSTANTIATE(CSCArrayAdapterBatch)
INSTANTIATE(SparsePageAdapterBatch)
INSTANTIATE(ColumnarAdapterBatch)
INSTANTIATE(EncColumnarAdapterBatch)

#undef INSTANTIATE

namespace {
/**
 * @brief A view over gathered sketch values.
 */
template <typename T>
struct QuantileAllreduce {
  common::Span<T> global_values;
  common::Span<bst_idx_t> worker_indptr;
  common::Span<bst_idx_t> feature_indptr;
  bst_feature_t n_features{0};
  /**
   * @brief Get sketch values of the a feature from a worker.
   *
   * @param rank rank of target worker
   * @param fidx feature idx
   */
  [[nodiscard]] auto Values(int32_t rank, bst_feature_t fidx) const {
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
}  // anonymous namespace

auto SketchContainerImpl::GatherSketchInfo(Context const *ctx, MetaInfo const &info,
                                           std::vector<WQSketch::SummaryContainer> const &reduced)
    -> std::tuple<std::vector<bst_idx_t>, std::vector<bst_idx_t>, std::vector<WQSketch::Entry>> {
  std::vector<bst_idx_t> worker_segments(1, 0);
  auto world = collective::GetWorldSize();
  auto rank = collective::GetRank();
  bst_feature_t n_columns = sketches_.size();

  // get the size of each feature.
  std::vector<bst_idx_t> sketch_size;
  sketch_size.reserve(reduced.size());
  size_t fidx{0};
  for (auto const &sketch : reduced) {
    if (IsCat(feature_types_, fidx)) {
      sketch_size.push_back(0);
    } else {
      sketch_size.push_back(sketch.Size());
    }
    ++fidx;
  }
  // Turn the size into CSC indptr
  std::vector<bst_idx_t> sketches_scan((n_columns + 1) * world, 0);
  size_t beg_scan = rank * (n_columns + 1);  // starting storage for current worker.
  std::partial_sum(sketch_size.cbegin(), sketch_size.cend(), sketches_scan.begin() + beg_scan + 1);

  // Gather all column pointers
  auto rc =
      collective::GlobalSum(ctx, info, linalg::MakeVec(sketches_scan.data(), sketches_scan.size()));
  if (!rc.OK()) {
    collective::SafeColl(collective::Fail("Failed to get sketch scan.", std::move(rc)));
  }

  for (int32_t i = 0; i < world; ++i) {
    size_t back = (i + 1) * (n_columns + 1) - 1;
    auto n_entries = sketches_scan.at(back);
    worker_segments.push_back(n_entries);
  }
  // Offset of sketch from each worker.
  std::partial_sum(worker_segments.begin(), worker_segments.end(), worker_segments.begin());
  CHECK_GE(worker_segments.size(), 1);
  auto total = worker_segments.back();

  std::vector<WQSketch::Entry> global_sketches(total, WQSketch::Entry{0, 0, 0, 0});
  auto worker_sketch = Span<WQSketch::Entry>{global_sketches}.subspan(
      worker_segments[rank], worker_segments[rank + 1] - worker_segments[rank]);
  auto cursor{worker_sketch.begin()};
  for (size_t fidx = 0; fidx < reduced.size(); ++fidx) {
    auto const &sketch = reduced[fidx];
    if (IsCat(feature_types_, fidx) || sketch.Empty()) {
      // Nothing to copy for categorical features or empty sketches.
      continue;
    }
    auto entries = sketch.Entries();
    cursor = std::copy(entries.cbegin(), entries.cend(), cursor);
  }

  static_assert(sizeof(WQSketch::Entry) / 4 == sizeof(float), "Unexpected size of sketch entry.");
  if (global_sketches.empty()) {
    return std::make_tuple(std::move(worker_segments), std::move(sketches_scan),
                           std::move(global_sketches));
  }
  rc = collective::GlobalSum(
      ctx, info,
      linalg::MakeVec(reinterpret_cast<float *>(global_sketches.data()),
                      global_sketches.size() * sizeof(WQSketch::Entry) / sizeof(float)));
  if (!rc.OK()) {
    collective::SafeColl(collective::Fail("Failed to get sketch.", std::move(rc)));
  }
  return std::make_tuple(std::move(worker_segments), std::move(sketches_scan),
                         std::move(global_sketches));
}

void SketchContainerImpl::AllreduceCategories(Context const *ctx, MetaInfo const &info) {
  auto world_size = collective::GetWorldSize();
  auto rank = collective::GetRank();
  if (world_size == 1 || info.IsColumnSplit()) {
    return;
  }

  // CSC indptr to each feature
  std::vector<size_t> feature_ptr;
  feature_ptr.reserve(categories_.size() + 1);
  feature_ptr.push_back(0);
  for (auto const &feat : categories_) {
    feature_ptr.push_back(feature_ptr.back() + feat.size());
  }
  CHECK_EQ(feature_ptr.front(), 0);

  // gather all feature ptrs from workers
  std::vector<bst_idx_t> global_feat_ptrs(feature_ptr.size() * world_size, 0);
  size_t feat_begin = rank * feature_ptr.size();  // pointer to current worker
  std::copy(feature_ptr.begin(), feature_ptr.end(), global_feat_ptrs.begin() + feat_begin);
  auto rc = collective::GlobalSum(
      ctx, info, linalg::MakeVec(global_feat_ptrs.data(), global_feat_ptrs.size()));

  // move all categories into a flatten vector to prepare for allreduce
  size_t total = feature_ptr.back();
  std::vector<float> flatten(total, 0);
  auto cursor{flatten.begin()};
  for (auto const &feat : categories_) {
    cursor = std::copy(feat.cbegin(), feat.cend(), cursor);
  }

  // indptr for indexing workers
  std::vector<bst_idx_t> global_worker_ptr(world_size + 1, 0);
  global_worker_ptr[rank + 1] = total;  // shift 1 to right for constructing the indptr
  rc = collective::GlobalSum(ctx, info,
                             linalg::MakeVec(global_worker_ptr.data(), global_worker_ptr.size()));
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
  rc = collective::GlobalSum(ctx, info,
                             linalg::MakeVec(global_categories.data(), global_categories.size()));
  QuantileAllreduce<float> allreduce_result{global_categories, global_worker_ptr, global_feat_ptrs,
                                            static_cast<bst_feature_t>(categories_.size())};
  ParallelFor(categories_.size(), n_threads_, [&](auto fidx) {
    if (!IsCat(feature_types_, fidx)) {
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
        categories_[fidx].emplace(c);
      }
    }
  });
}

auto SketchContainerImpl::AllReduce(Context const *ctx, MetaInfo const &info)
    -> std::vector<WQSketch::SummaryContainer> {
  monitor_.Start(__func__);

  // Sanity check the number of features across workers before allreduce
  bst_feature_t n_columns = sketches_.size();
  auto rc = collective::Allreduce(ctx, &n_columns, collective::Op::kMax);
  collective::SafeColl(rc);
  CHECK_EQ(n_columns, sketches_.size()) << "Number of columns differs across workers";

  AllreduceCategories(ctx, info);

  std::vector<bst_feature_t> numeric_features;
  numeric_features.reserve(n_columns);
  for (bst_feature_t fidx = 0; fidx < n_columns; ++fidx) {
    if (!IsCat(feature_types_, fidx)) {
      numeric_features.push_back(fidx);
    }
  }

  std::vector<WQSketch::SummaryContainer> reduced(sketches_.size());

  // Cap the per-feature summary size during local and distributed merge.
  auto const max_cut_target = static_cast<std::size_t>(max_bins_ * WQSketch::kFactor);
  ParallelFor(numeric_features.size(), n_threads_, [&](size_t idx) {
    auto fidx = numeric_features[idx];
    reduced[fidx] = sketches_[fidx].GetSummary(max_cut_target);
  });

  // Early exit: no allreduce needed when one worker, column-split, or no numeric features.
  if (collective::GetWorldSize() == 1 || info.IsColumnSplit() || numeric_features.empty()) {
    monitor_.Stop(__func__);
    return std::move(reduced);
  }

  // Serialize local sketches to a byte array for allreduce
  auto merged = SketchReducePayload::SerializeFromSummaries(
      Span<bst_feature_t const>{numeric_features}, reduced);
  WQSketch::SummaryContainer tmp;
  tmp.Reserve(max_cut_target * 2);  // workspace for merging sketches during allreduce
  auto reduce_rc = collective::ReduceV(
      ctx, &merged,
      [&](common::Span<std::byte const> a, common::Span<std::byte const> b,
          std::vector<std::byte> *out) {
        auto a_payload = SketchReducePayload::Parse(a);
        auto b_payload = SketchReducePayload::Parse(b);

        auto max_entries = a_payload.TotalEntries() + b_payload.TotalEntries();
        auto max_pruned_entries = max_cut_target * numeric_features.size();
        max_entries = std::min(max_entries, max_pruned_entries);
        SketchReducePayload::InitHeader(out, numeric_features.size(), max_entries);

        for (std::size_t i = 0; i < numeric_features.size(); ++i) {
          auto a_summary = a_payload.SummaryAt(i);
          auto b_summary = b_payload.SummaryAt(i);
          tmp.CopyFrom(a_summary);
          tmp.SetCombine(b_summary);
          tmp.SetPrune(max_cut_target);

          auto pruned_entries = tmp.Entries();
          SketchReducePayload::AppendEntries(out, i, pruned_entries);
        }
      });
  collective::SafeColl(reduce_rc);

  // Deserialize the sketches back to summary containers.
  auto reduced_payload = SketchReducePayload::Parse(Span<std::byte const>{merged});
  CHECK_EQ(reduced_payload.NumFeatures(), numeric_features.size());
  for (std::size_t i = 0; i < numeric_features.size(); ++i) {
    auto fidx = numeric_features[i];
    auto entries = reduced_payload.Entries(i);
    auto n_entries = entries.size();

    reduced[fidx].Reserve(n_entries);
    reduced[fidx].CopyFrom(WQSketch::Summary{entries, n_entries});
  }
  monitor_.Stop(__func__);
  return std::move(reduced);
}

template <typename SketchType>
void AddCutPoint(typename SketchType::SummaryContainer const &summary, int max_bin,
                 HistogramCuts *cuts) {
  size_t required_cuts = std::min(summary.Size(), static_cast<size_t>(max_bin));
  auto &cut_values = cuts->cut_values_.HostVector();
  auto const entries = summary.Entries();
  // Use raw pointer in the cut extraction loop to avoid per-access bounds checks.
  auto const *summary_data = entries.data();
  // we use the min_value as the first (0th) element, hence starting from 1.
  for (size_t i = 1; i < required_cuts; ++i) {
    bst_float cpt = summary_data[i].value;
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
  // With column-wise data split, the categories may be empty.
  auto max_cat =
      categories.empty() ? 0.0f : *std::max_element(categories.cbegin(), categories.cend());
  CheckMaxCat(max_cat, categories.size());
  for (bst_cat_t i = 0; i <= AsCat(max_cat); ++i) {
    cut_values.push_back(i);
  }
  return max_cat;
}

void SketchContainerImpl::MakeCuts(Context const *ctx, MetaInfo const &info,
                                   HistogramCuts *p_cuts) {
  monitor_.Start(__func__);
  auto reduced = this->AllReduce(ctx, info);

  p_cuts->min_vals_.HostVector().resize(sketches_.size(), 0.0f);
  std::vector<WQSketch::SummaryContainer> final_summaries(reduced.size());

  ParallelFor(reduced.size(), n_threads_, Sched::Guided(), [&](size_t fidx) {
    if (IsCat(feature_types_, fidx)) {
      return;
    }
    WQSketch::SummaryContainer &a = final_summaries[fidx];
    size_t max_num_bins = std::min(reduced[fidx].Size(), static_cast<size_t>(max_bins_));
    a.Reserve(std::max(reduced[fidx].Size(), max_num_bins + 1));
    CHECK(a.Entries().data());
    if (!reduced[fidx].Empty()) {
      a.CopyFrom(reduced[fidx]);
      a.SetPrune(max_num_bins + 1);
      auto const a_entries = a.Entries();
      auto const reduced_entries = reduced[fidx].Entries();
      CHECK(a_entries.data() && reduced_entries.data());
      CHECK(!a_entries.empty());
      const bst_float mval = a_entries.front().value;
      p_cuts->min_vals_.HostVector()[fidx] = mval - fabs(mval) - 1e-5f;
    } else {
      // Empty column.
      const float mval = 1e-5f;
      p_cuts->min_vals_.HostVector()[fidx] = mval;
    }
  });

  float max_cat{-1.f};
  for (size_t fid = 0; fid < reduced.size(); ++fid) {
    size_t max_num_bins = std::min(reduced[fid].Size(), static_cast<size_t>(max_bins_));
    WQSketch::SummaryContainer const &a = final_summaries[fid];
    if (IsCat(feature_types_, fid)) {
      max_cat = std::max(max_cat, AddCategories(categories_.at(fid), p_cuts));
    } else {
      AddCutPoint<WQSketch>(a, max_num_bins, p_cuts);
      // push a value that is greater than anything
      auto const a_entries = a.Entries();
      const bst_float cpt =
          !a_entries.empty() ? a_entries.back().value : p_cuts->min_vals_.HostVector()[fid];
      // this must be bigger than last value in a scale
      const bst_float last = cpt + (fabs(cpt) + 1e-5f);
      p_cuts->cut_values_.HostVector().push_back(last);
    }

    // Ensure that every feature gets at least one quantile point
    CHECK_LE(p_cuts->cut_values_.HostVector().size(), std::numeric_limits<uint32_t>::max());
    auto cut_size = static_cast<uint32_t>(p_cuts->cut_values_.HostVector().size());
    CHECK_GT(cut_size, p_cuts->cut_ptrs_.HostVector().back());
    p_cuts->cut_ptrs_.HostVector().push_back(cut_size);
  }

  p_cuts->SetCategorical(this->has_categorical_, max_cat);
  monitor_.Stop(__func__);
}

HostSketchContainer::HostSketchContainer(Context const *ctx, bst_bin_t max_bins,
                                         common::Span<FeatureType const> ft,
                                         std::vector<bst_idx_t> columns_size, bool use_group)
    : SketchContainerImpl{ctx, columns_size, max_bins, ft, use_group} {
  monitor_.Init(__func__);
  ParallelFor(sketches_.size(), n_threads_, Sched::Auto(), [&](auto i) {
    auto n_bins = std::min(static_cast<bst_idx_t>(max_bins_), columns_size_[i]);
    n_bins = std::max(n_bins, static_cast<decltype(n_bins)>(1));
    auto eps = 1.0 / (static_cast<float>(n_bins) * WQSketch::kFactor);
    if (!IsCat(this->feature_types_, i)) {
      sketches_[i] = WQSketch{columns_size_[i], eps};
    }
  });
}

void SortedSketchContainer::PushColPage(SparsePage const &page, MetaInfo const &info,
                                        Span<float const> hessian) {
  monitor_.Start(__func__);
  // glue these conditions using ternary operator to avoid making data copies.
  auto const &weights =
      hessian.empty() ? (use_group_ind_ ? detail::UnrollGroupWeights(info)  // use group weight
                                        : info.weights_.HostVector())       // use sample weight
                      : MergeWeights(info, hessian, use_group_ind_,
                                     n_threads_);  // use hessian merged with group/sample weights
  CHECK_EQ(weights.size(), info.num_row_);

  auto view = page.GetView();
  ParallelFor(view.Size(), n_threads_, [&](size_t fidx) {
    auto column = view[fidx];
    if (IsCat(feature_types_, fidx)) {
      for (auto c : column) {
        categories_[fidx].emplace(c.fvalue);
      }
      return;
    }
    sketches_[fidx].PushSorted(column, weights, static_cast<size_t>(max_bins_));
  });
  monitor_.Stop(__func__);
}

}  // namespace xgboost::common
