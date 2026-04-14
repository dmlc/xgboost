/**
 * Copyright 2020-2025, XGBoost Contributors
 */
#include "quantile.h"

#include <cstddef>  // for byte
#include <cstdint>  // for uint64_t
#include <iterator>
#include <limits>
#include <type_traits>  // for is_trivially_copyable_v
#include <utility>

#include "../collective/aggregator.h"
#include "../common/error_msg.h"  // for InvalidMaxBin
#include "../data/adapter.h"
#include "categorical.h"
#include "hist_util.h"

namespace xgboost::common {
HostSketchContainer::HostSketchContainer(Context const *ctx, bst_bin_t max_bin,
                                         Span<FeatureType const> feature_types,
                                         std::vector<bst_idx_t> columns_size, bool use_group)
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
  ParallelFor(sketches_.size(), n_threads_, Sched::Auto(), [&](auto i) {
    auto eps = SketchEpsilon(max_bins_, columns_size_[i]);
    if (!IsCat(this->feature_types_, i)) {
      sketches_[i] = WQSketch{columns_size_[i], eps};
    }
  });
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
      while (cur_group + 1 < group_ptr.size() && i >= group_ptr[cur_group + 1]) {
        ++cur_group;
      }
      results[i] = hessian[i] * get_weight(cur_group);
    }
  } else {
    ParallelFor(hessian.size(), n_threads, Sched::Auto(),
                [&](auto i) { results[i] = hessian[i] * get_weight(i); });
  }
  return results;
}

template <typename T>
void WritePODAt(std::vector<std::byte> *out, std::size_t offset, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  auto const *src = reinterpret_cast<std::byte const *>(&value);
  std::copy_n(src, sizeof(T), out->begin() + static_cast<std::ptrdiff_t>(offset));
}

template <typename T>
[[nodiscard]] T ReadPOD(Span<std::byte const> bytes, std::size_t *cursor) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  CHECK_LE(*cursor, bytes.size());
  CHECK_LE(sizeof(T), bytes.size() - *cursor);
  auto *dst = reinterpret_cast<std::byte *>(&value);
  std::copy_n(bytes.data() + *cursor, sizeof(T), dst);
  *cursor += sizeof(T);
  return value;
}

// Serialization payload for distributed numerical sketch merging over AllreduceV.
// Encodes per-feature represented element counts, per-feature entry counts, and contiguous
// sketch entries.
struct SketchReducePayload {
  [[nodiscard]] static std::vector<std::byte> SerializeFromSummaries(
      Span<bst_feature_t const> numeric_features,
      std::vector<WQuantileSketch::SummaryContainer> const &reduced,
      std::vector<std::size_t> const &num_elements) {
    CHECK_EQ(reduced.size(), num_elements.size());
    std::size_t total_entries = 0;
    for (auto fidx : numeric_features) {
      total_entries += reduced.at(fidx).Size();
    }

    std::vector<std::byte> bytes;
    InitHeader(&bytes, numeric_features.size(), total_entries);

    for (std::size_t i = 0; i < numeric_features.size(); ++i) {
      auto fidx = numeric_features[i];
      SetNumElements(&bytes, numeric_features.size(), i, num_elements.at(fidx));
      auto out_entries = reduced.at(fidx).Entries();
      AppendEntries(&bytes, numeric_features.size(), i, out_entries);
    }
    auto header_bytes = HeaderBytes(numeric_features.size());
    CHECK_EQ((bytes.size() - header_bytes) / sizeof(WQuantileSketch::Entry), total_entries);
    return bytes;
  }

  [[nodiscard]] static std::size_t HeaderBytes(std::size_t n_features) {
    return sizeof(std::uint64_t) + 2 * n_features * sizeof(std::uint64_t);
  }

  static void SetNumElements(std::vector<std::byte> *bytes, std::size_t n_features, std::size_t i,
                             std::size_t num_elements) {
    CHECK(bytes);
    auto count_offset = sizeof(std::uint64_t) + i * sizeof(std::uint64_t);
    CHECK_LE(count_offset + sizeof(std::uint64_t), HeaderBytes(n_features));
    WritePODAt<std::uint64_t>(bytes, count_offset, static_cast<std::uint64_t>(num_elements));
  }

  static void AppendEntries(std::vector<std::byte> *bytes, std::size_t n_features, std::size_t i,
                            Span<WQuantileSketch::Entry const> entries) {
    CHECK(bytes);
    auto count_offset =
        sizeof(std::uint64_t) + n_features * sizeof(std::uint64_t) + i * sizeof(std::uint64_t);
    CHECK_LE(count_offset + sizeof(std::uint64_t), HeaderBytes(n_features));
    WritePODAt<std::uint64_t>(bytes, count_offset, static_cast<std::uint64_t>(entries.size()));
    if (entries.empty()) {
      return;
    }
    auto entries_bytes = entries.size() * sizeof(WQuantileSketch::Entry);
    auto const *src = reinterpret_cast<std::byte const *>(entries.data());
    bytes->insert(bytes->end(), src, src + entries_bytes);
  }

  static void InitHeader(std::vector<std::byte> *bytes, std::size_t n_features,
                         std::size_t max_entries) {
    CHECK(bytes);
    auto const header_bytes = HeaderBytes(n_features);
    bytes->clear();
    bytes->reserve(header_bytes + max_entries * sizeof(WQuantileSketch::Entry));
    bytes->resize(header_bytes);
    WritePODAt<std::uint64_t>(bytes, 0, static_cast<std::uint64_t>(n_features));
  }

  [[nodiscard]] static SketchReducePayload Parse(Span<std::byte> bytes) {
    std::size_t cursor = 0;
    auto n_features = ReadPOD<std::uint64_t>(bytes, &cursor);

    std::vector<std::size_t> num_elements(n_features, 0);
    for (std::size_t i = 0; i < n_features; ++i) {
      num_elements[i] = static_cast<std::size_t>(ReadPOD<std::uint64_t>(bytes, &cursor));
    }

    std::vector<std::size_t> offsets(n_features + 1, 0);
    for (std::size_t i = 0; i < n_features; ++i) {
      auto n_i = static_cast<std::size_t>(ReadPOD<std::uint64_t>(bytes, &cursor));
      offsets[i + 1] = offsets[i] + n_i;
    }

    auto n_entries = offsets.back();
    auto payload_bytes = n_entries * sizeof(WQuantileSketch::Entry);
    CHECK_EQ(cursor + payload_bytes, bytes.size());

    WQuantileSketch::Entry *entries = nullptr;
    if (n_entries != 0) {
      auto ptr = bytes.data() + cursor;
      auto addr = reinterpret_cast<std::uintptr_t>(ptr);
      CHECK_EQ(addr % alignof(WQuantileSketch::Entry), 0);
      entries = reinterpret_cast<WQuantileSketch::Entry *>(ptr);
    }

    return {std::move(offsets), std::move(num_elements),
            Span<WQuantileSketch::Entry>{entries, n_entries}};
  }

  [[nodiscard]] std::size_t NumFeatures() const { return offsets_.size() - 1; }
  [[nodiscard]] std::size_t TotalEntries() const { return entries_.size(); }
  [[nodiscard]] std::size_t NumElements(std::size_t idx) const { return num_elements_.at(idx); }

  [[nodiscard]] Span<WQuantileSketch::Entry> Entries(std::size_t idx) const {
    auto beg = offsets_.at(idx);
    auto end = offsets_.at(idx + 1);
    auto n = end - beg;
    if (n == 0) {
      return Span<WQuantileSketch::Entry>{};
    }
    return {entries_.data() + beg, n};
  }

  [[nodiscard]] WQuantileSketch::Summary SummaryAt(std::size_t idx) const {
    auto entries = this->Entries(idx);
    return {entries, entries.size()};
  }

 private:
  SketchReducePayload(std::vector<std::size_t> offsets, std::vector<std::size_t> num_elements,
                      Span<WQuantileSketch::Entry> entries)
      : offsets_{std::move(offsets)}, num_elements_{std::move(num_elements)}, entries_{entries} {}

  std::vector<std::size_t> offsets_;
  std::vector<std::size_t> num_elements_;
  Span<WQuantileSketch::Entry> entries_;
};

// Serialization payload for distributed categorical value union over AllreduceV.
// Encodes per-feature value counts plus contiguous category values.
struct CategoricalReducePayload {
  [[nodiscard]] static std::vector<std::byte> SerializeFromCategories(
      Span<bst_feature_t const> categorical_features,
      std::vector<std::set<float>> const &categories) {
    std::size_t total_values = 0;
    for (auto fidx : categorical_features) {
      total_values += categories.at(fidx).size();
    }

    std::vector<std::byte> bytes;
    InitHeader(&bytes, categorical_features.size(), total_values);
    for (std::size_t i = 0; i < categorical_features.size(); ++i) {
      auto fidx = categorical_features[i];
      AppendValues(&bytes, i, categories.at(fidx));
    }

    auto header_bytes = HeaderBytes(categorical_features.size());
    CHECK_EQ((bytes.size() - header_bytes) / sizeof(float), total_values);
    return bytes;
  }

  [[nodiscard]] static std::size_t HeaderBytes(std::size_t n_features) {
    return sizeof(std::uint64_t) + n_features * sizeof(std::uint64_t);
  }

  static void AppendValues(std::vector<std::byte> *bytes, std::size_t i, Span<float const> values) {
    CHECK(bytes);
    auto count_offset = sizeof(std::uint64_t) + i * sizeof(std::uint64_t);
    CHECK_LE(count_offset + sizeof(std::uint64_t), bytes->size());
    WritePODAt<std::uint64_t>(bytes, count_offset, static_cast<std::uint64_t>(values.size()));
    if (values.empty()) {
      return;
    }
    auto values_bytes = values.size() * sizeof(float);
    auto const *src = reinterpret_cast<std::byte const *>(values.data());
    bytes->insert(bytes->end(), src, src + values_bytes);
  }

  static void AppendValues(std::vector<std::byte> *bytes, std::size_t i,
                           std::set<float> const &values) {
    CHECK(bytes);
    auto count_offset = sizeof(std::uint64_t) + i * sizeof(std::uint64_t);
    CHECK_LE(count_offset + sizeof(std::uint64_t), bytes->size());
    WritePODAt<std::uint64_t>(bytes, count_offset, static_cast<std::uint64_t>(values.size()));
    if (values.empty()) {
      return;
    }

    auto offset = bytes->size();
    bytes->resize(offset + values.size() * sizeof(float));
    auto dst = bytes->begin() + static_cast<std::ptrdiff_t>(offset);
    for (auto value : values) {
      auto const *src = reinterpret_cast<std::byte const *>(&value);
      dst = std::copy_n(src, sizeof(float), dst);
    }
  }

  static void InitHeader(std::vector<std::byte> *bytes, std::size_t n_features,
                         std::size_t max_values) {
    CHECK(bytes);
    auto const header_bytes = HeaderBytes(n_features);
    bytes->clear();
    bytes->reserve(header_bytes + max_values * sizeof(float));
    bytes->resize(header_bytes);
    WritePODAt<std::uint64_t>(bytes, 0, static_cast<std::uint64_t>(n_features));
  }

  [[nodiscard]] static CategoricalReducePayload Parse(Span<std::byte> bytes) {
    std::size_t cursor = 0;
    auto n_features = ReadPOD<std::uint64_t>(bytes, &cursor);

    std::vector<std::size_t> offsets(n_features + 1, 0);
    for (std::size_t i = 0; i < n_features; ++i) {
      auto n_i = static_cast<std::size_t>(ReadPOD<std::uint64_t>(bytes, &cursor));
      offsets[i + 1] = offsets[i] + n_i;
    }

    auto n_values = offsets.back();
    auto payload_bytes = n_values * sizeof(float);
    CHECK_EQ(cursor + payload_bytes, bytes.size());

    float const *values = nullptr;
    if (n_values != 0) {
      auto ptr = bytes.data() + cursor;
      auto addr = reinterpret_cast<std::uintptr_t>(ptr);
      CHECK_EQ(addr % alignof(float), 0);
      values = reinterpret_cast<float const *>(ptr);
    }

    return {std::move(offsets), Span<float const>{values, n_values}};
  }

  [[nodiscard]] std::size_t NumFeatures() const { return offsets_.size() - 1; }
  [[nodiscard]] std::size_t TotalValues() const { return values_.size(); }

  [[nodiscard]] Span<float const> Values(std::size_t idx) const {
    auto beg = offsets_.at(idx);
    auto end = offsets_.at(idx + 1);
    auto n = end - beg;
    if (n == 0) {
      return Span<float const>{};
    }
    return {values_.data() + beg, n};
  }

 private:
  CategoricalReducePayload(std::vector<std::size_t> offsets, Span<float const> values)
      : offsets_{std::move(offsets)}, values_{values} {}

  std::vector<std::size_t> offsets_;
  Span<float const> values_;
};
}  // anonymous namespace

void HostSketchContainer::PushRowPage(SparsePage const &page, MetaInfo const &info,
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

auto HostSketchContainer::AllreduceCategories(Context const *ctx, MetaInfo const &info,
                                              Span<bst_feature_t const> categorical_features)
    -> std::vector<std::set<float>> {
  std::vector<std::set<float>> reduced_categories(categorical_features.size());
  if (categorical_features.empty()) {
    return reduced_categories;
  }

  if (collective::GetWorldSize() == 1 || info.IsColumnSplit()) {
    for (std::size_t i = 0; i < categorical_features.size(); ++i) {
      reduced_categories[i] = categories_[categorical_features[i]];
    }
    return reduced_categories;
  }

  auto merged =
      CategoricalReducePayload::SerializeFromCategories(categorical_features, categories_);
  std::vector<float> merge_workspace;
  auto rc = collective::AllreduceV(
      ctx, &merged,
      [&](common::Span<std::byte const> a, common::Span<std::byte const> b,
          std::vector<std::byte> *out) {
        auto a_payload = CategoricalReducePayload::Parse(
            Span<std::byte>{const_cast<std::byte *>(a.data()), a.size()});
        auto b_payload = CategoricalReducePayload::Parse(
            Span<std::byte>{const_cast<std::byte *>(b.data()), b.size()});
        CHECK_EQ(a_payload.NumFeatures(), categorical_features.size());
        CHECK_EQ(b_payload.NumFeatures(), categorical_features.size());

        auto max_values = a_payload.TotalValues() + b_payload.TotalValues();
        CategoricalReducePayload::InitHeader(out, categorical_features.size(), max_values);

        for (std::size_t i = 0; i < categorical_features.size(); ++i) {
          auto a_values = a_payload.Values(i);
          auto b_values = b_payload.Values(i);
          merge_workspace.clear();
          merge_workspace.reserve(a_values.size() + b_values.size());
          std::set_union(a_values.cbegin(), a_values.cend(), b_values.cbegin(), b_values.cend(),
                         std::back_inserter(merge_workspace));
          CategoricalReducePayload::AppendValues(out, i, Span<float const>{merge_workspace});
        }
      });
  collective::SafeColl(rc);

  auto reduced_payload = CategoricalReducePayload::Parse(Span<std::byte>{merged});
  CHECK_EQ(reduced_payload.NumFeatures(), categorical_features.size());
  for (std::size_t i = 0; i < categorical_features.size(); ++i) {
    auto values = reduced_payload.Values(i);
    reduced_categories[i].insert(values.cbegin(), values.cend());
  }
  return reduced_categories;
}

auto HostSketchContainer::AllReduce(Context const *ctx, MetaInfo const &info,
                                    Span<bst_feature_t const> numeric_features)
    -> std::vector<WQSketch::SummaryContainer> {
  monitor_.Start(__func__);

  // Sanity check the number of features across workers before allreduce
  bst_feature_t n_columns = sketches_.size();
  auto rc = collective::Allreduce(ctx, &n_columns, collective::Op::kMax);
  collective::SafeColl(rc);
  CHECK_EQ(n_columns, sketches_.size()) << "Number of columns differs across workers";

  std::vector<WQSketch::SummaryContainer> reduced(sketches_.size());
  std::vector<std::size_t> num_elements(sketches_.size(), 0);

  // Size local summaries with the same O(log n / eps) budget as the single-machine sketch.
  ParallelFor(numeric_features.size(), n_threads_, [&](size_t idx) {
    auto fidx = numeric_features[idx];
    num_elements[fidx] = sketches_[fidx].NumElements();
    auto cut_target = SketchSummaryBudget(max_bins_, num_elements[fidx]);
    reduced[fidx] = sketches_[fidx].GetSummary(cut_target);
  });

  // Early exit: no allreduce needed when one worker, column-split, or no numeric features.
  if (collective::GetWorldSize() == 1 || info.IsColumnSplit() || numeric_features.empty()) {
    monitor_.Stop(__func__);
    return reduced;
  }

  // Serialize local sketches to a byte array for allreduce
  auto merged = SketchReducePayload::SerializeFromSummaries(
      Span<bst_feature_t const>{numeric_features}, reduced, num_elements);
  WQSketch::SummaryContainer tmp;
  auto reduce_rc = collective::AllreduceV(
      ctx, &merged,
      [&](common::Span<std::byte const> a, common::Span<std::byte const> b,
          std::vector<std::byte> *out) {
        auto a_payload = SketchReducePayload::Parse(
            Span<std::byte>{const_cast<std::byte *>(a.data()), a.size()});
        auto b_payload = SketchReducePayload::Parse(
            Span<std::byte>{const_cast<std::byte *>(b.data()), b.size()});
        CHECK_EQ(a_payload.NumFeatures(), numeric_features.size());
        CHECK_EQ(b_payload.NumFeatures(), numeric_features.size());

        auto max_entries = a_payload.TotalEntries() + b_payload.TotalEntries();
        std::size_t max_pruned_entries{0};
        for (std::size_t i = 0; i < numeric_features.size(); ++i) {
          auto num_elements = a_payload.NumElements(i) + b_payload.NumElements(i);
          max_pruned_entries += SketchSummaryBudget(max_bins_, num_elements);
        }
        max_entries = std::min(max_entries, max_pruned_entries);
        SketchReducePayload::InitHeader(out, numeric_features.size(), max_entries);

        for (std::size_t i = 0; i < numeric_features.size(); ++i) {
          auto a_summary = a_payload.SummaryAt(i);
          auto b_summary = b_payload.SummaryAt(i);
          auto num_elements = a_payload.NumElements(i) + b_payload.NumElements(i);
          auto cut_target = SketchSummaryBudget(max_bins_, num_elements);
          tmp.Reserve(a_summary.Size() + b_summary.Size());
          tmp.CopyFrom(a_summary);
          tmp.SetCombine(b_summary);
          tmp.SetPrune(cut_target);

          SketchReducePayload::SetNumElements(out, numeric_features.size(), i, num_elements);
          auto pruned_entries = tmp.Entries();
          SketchReducePayload::AppendEntries(out, numeric_features.size(), i, pruned_entries);
        }
      });
  collective::SafeColl(reduce_rc);

  // Deserialize the sketches back to summary containers.
  auto reduced_payload = SketchReducePayload::Parse(Span<std::byte>{merged});
  CHECK_EQ(reduced_payload.NumFeatures(), numeric_features.size());
  for (std::size_t i = 0; i < numeric_features.size(); ++i) {
    auto fidx = numeric_features[i];
    auto entries = reduced_payload.Entries(i);
    auto n_entries = entries.size();

    reduced[fidx].Reserve(n_entries);
    reduced[fidx].CopyFrom(WQSketch::Summary{entries, n_entries});
  }
  monitor_.Stop(__func__);
  return reduced;
}

void AddCutPoints(WQSummaryContainer const &summary, size_t max_bin, HistogramCuts *cuts) {
  auto &cut_values = cuts->cut_values_.HostVector();
  auto queried = summary.QueryCutValues(max_bin);
  cut_values.insert(cut_values.end(), queried.cbegin(), queried.cend());
}

void AddCategories(std::set<float> const &categories, float *max_cat, HistogramCuts *cuts) {
  if (std::any_of(categories.cbegin(), categories.cend(), InvalidCat)) {
    InvalidCategory();
  }
  auto &cut_values = cuts->cut_values_.HostVector();
  // With column-wise data split, the categories may be empty.
  auto feature_max_cat =
      categories.empty() ? 0.0f : *std::max_element(categories.cbegin(), categories.cend());
  CheckMaxCat(feature_max_cat, categories.size());
  *max_cat = std::max(*max_cat, feature_max_cat);
  for (bst_cat_t i = 0; i <= AsCat(feature_max_cat); ++i) {
    cut_values.push_back(i);
  }
}

HistogramCuts HostSketchContainer::MakeCuts(Context const *ctx, MetaInfo const &info) {
  monitor_.Start(__func__);
  HistogramCuts cuts{static_cast<bst_feature_t>(sketches_.size())};
  auto *p_cuts = &cuts;

  std::vector<bst_feature_t> numeric_features;
  std::vector<bst_feature_t> categorical_features;
  numeric_features.reserve(sketches_.size());
  categorical_features.reserve(sketches_.size());
  for (bst_feature_t fidx = 0; fidx < sketches_.size(); ++fidx) {
    if (IsCat(feature_types_, fidx)) {
      categorical_features.push_back(fidx);
    } else {
      numeric_features.push_back(fidx);
    }
  }

  auto reduced_numerical = this->AllReduce(ctx, info, Span<bst_feature_t const>{numeric_features});
  auto reduced_categories =
      this->AllreduceCategories(ctx, info, Span<bst_feature_t const>{categorical_features});
  std::vector<std::size_t> categorical_index(sketches_.size(), 0);
  for (std::size_t i = 0; i < categorical_features.size(); ++i) {
    categorical_index[categorical_features[i]] = i;
  }

  auto &h_cut_ptrs = p_cuts->cut_ptrs_.HostVector();
  float max_cat{-1.f};
  for (size_t fid = 0; fid < reduced_numerical.size(); ++fid) {
    size_t max_num_bins = std::min(reduced_numerical[fid].Size(), static_cast<size_t>(max_bins_));
    if (IsCat(feature_types_, fid)) {
      AddCategories(reduced_categories[categorical_index[fid]], &max_cat, p_cuts);
    } else {
      AddCutPoints(reduced_numerical[fid], max_num_bins, p_cuts);
    }

    // Ensure that every feature gets at least one quantile point
    CHECK_LE(p_cuts->cut_values_.HostVector().size(), std::numeric_limits<uint32_t>::max());
    auto cut_size = static_cast<uint32_t>(p_cuts->cut_values_.HostVector().size());
    CHECK_GT(cut_size, h_cut_ptrs[fid]);
    h_cut_ptrs[fid + 1] = cut_size;
  }

  p_cuts->SetCategorical(this->has_categorical_, max_cat);
  monitor_.Stop(__func__);
  return cuts;
}

void HostSketchContainer::PushColPage(SparsePage const &page, MetaInfo const &info,
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
