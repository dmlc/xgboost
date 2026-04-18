/**
 * Copyright 2014-2026, XGBoost Contributors
 * \file quantile.h
 * \brief util to compute quantiles
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_QUANTILE_H_
#define XGBOOST_COMMON_QUANTILE_H_

#include <xgboost/data.h>
#include <xgboost/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "categorical.h"
#include "common.h"
#include "error_msg.h"        // GroupWeight
#include "optional_weight.h"  // OptionalWeights
#include "threading_utils.h"
#include "timer.h"

namespace xgboost::common {
/*!
 * \brief experimental wsummary
 * \tparam DType type of data content
 * \tparam RType type of rank
 */
template <typename DType = bst_float, typename RType = bst_float>
struct WQSummary {
  /*! \brief an entry in the sketch summary */
  struct Entry {
    /*! \brief minimum rank */
    RType rmin{};
    /*! \brief maximum rank */
    RType rmax{};
    /*! \brief maximum weight */
    RType wmin{};
    /*! \brief the value of data */
    DType value{};
    // constructor
    XGBOOST_DEVICE Entry() {}  // NOLINT
    // constructor
    XGBOOST_DEVICE Entry(RType rmin, RType rmax, RType wmin, DType value)
        : rmin(rmin), rmax(rmax), wmin(wmin), value(value) {}
    /*! \return rmin estimation for v strictly bigger than value */
    XGBOOST_DEVICE RType RMinNext() const { return rmin + wmin; }
    /*! \return rmax estimation for v strictly smaller than value */
    XGBOOST_DEVICE RType RMaxPrev() const { return rmax - wmin; }
  };
  // constructor
  WQSummary(Span<Entry> data, size_t current_elements)
      : data_{data}, current_elements_{current_elements} {}
  /*! \brief Return the number of valid entries in this summary. */
  [[nodiscard]] size_t Size() const { return current_elements_; }
  /*! \brief Return true if this summary has no valid entries. */
  [[nodiscard]] bool Empty() const { return this->Size() == 0; }
  /*! \brief Return a const span over valid entries [0, Size()). */
  [[nodiscard]] Span<Entry const> Entries() const { return {data_.data(), current_elements_}; }
  /*! \brief Set the number of valid entries in this summary. */
  void SetSize(size_t n) {
    CHECK_LE(n, data_.size());
    current_elements_ = n;
  }
  /*! \brief Clear summary contents while keeping allocated storage. */
  void Clear() { current_elements_ = 0; }
  /*!
   * \brief copy content from src
   * \param src source sketch
   */
  void CopyFrom(const WQSummary &src) {
    if (!src.data_.data()) {
      CHECK_EQ(src.current_elements_, 0);
      this->Clear();
      return;
    }
    if (!data_.data()) {
      CHECK_EQ(this->current_elements_, 0);
      CHECK_EQ(src.current_elements_, 0);
      return;
    }
    current_elements_ = src.current_elements_;
    std::copy(src.data_.data(), src.data_.data() + current_elements_, data_.data());
  }

  void SetFromSorted(std::vector<std::pair<DType, RType>> const &queue) {
    this->Clear();
    RType wsum = 0;
    for (size_t i = 0; i < queue.size();) {
      size_t j = i + 1;
      RType w = queue[i].second;
      while (j < queue.size() && queue[j].first == queue[i].first) {
        w += queue[j].second;
        ++j;
      }
      data_[current_elements_++] = Entry{wsum, wsum + w, w, queue[i].first};
      wsum += w;
      i = j;
    }
  }

  /*!
   * \brief Set this summary from sorted column entries and prune by max_size.
   *
   * The input column must be sorted by feature value.
   */
  void SetPruneSorted(common::Span<::xgboost::Entry const> column,
                      std::vector<float> const &weights, size_t max_size) {
    CHECK_GE(max_size, 1);
    CHECK_GE(data_.size(), max_size + 1);

    this->Clear();
    auto const *col_data = column.data();
    auto const col_size = column.size();
    double sum_total{0.0};
    std::size_t unique_values{0};
    double rmin{0.0};
    double wmin{0.0};
    bst_float last_fvalue{0.0f};
    double next_goal{-1.0f};

    // first pass
    for (size_t i = 0; i < col_size; ++i) {
      if (i == 0 || col_data[i - 1].fvalue != col_data[i].fvalue) {
        ++unique_values;
      }
      auto const &c = col_data[i];
      sum_total += weights[c.index];
    }

    if (unique_values <= max_size) {
      // When we have enough budget to keep every unique feature value, emit the exact
      // weighted summary instead of running the weighted goal-selection logic below.
      for (size_t i = 0; i < col_size; ++i) {
        auto const &c = col_data[i];
        if (i == 0) {
          last_fvalue = c.fvalue;
          wmin = weights[c.index];
          continue;
        }
        if (last_fvalue == c.fvalue) {
          wmin += weights[c.index];
          continue;
        }

        auto rmax = rmin + wmin;
        data_[this->Size()] = Entry(static_cast<bst_float>(rmin), static_cast<bst_float>(rmax),
                                    static_cast<bst_float>(wmin), last_fvalue);
        this->SetSize(this->Size() + 1);
        rmin = rmax;
        last_fvalue = c.fvalue;
        wmin = weights[c.index];
      }

      if (col_size != 0) {
        auto rmax = rmin + wmin;
        data_[this->Size()] = Entry(static_cast<bst_float>(rmin), static_cast<bst_float>(rmax),
                                    static_cast<bst_float>(wmin), last_fvalue);
        this->SetSize(this->Size() + 1);
      }
      return;
    }

    // second pass
    for (size_t i = 0; i < col_size; ++i) {
      auto const &c = col_data[i];
      if (next_goal == -1.0f) {
        next_goal = 0.0f;
        last_fvalue = c.fvalue;
        wmin = weights[c.index];
        continue;
      }
      if (last_fvalue != c.fvalue) {
        double rmax = rmin + wmin;
        auto summary_size = this->Size();
        if (rmax >= next_goal && summary_size != max_size) {
          if (summary_size == 0 || last_fvalue > data_[summary_size - 1].value) {
            CHECK_LT(summary_size, max_size) << "invalid maximum size max_size=" << max_size
                                             << ", stemp.current_elements=" << summary_size;
            data_[summary_size] = Entry(static_cast<bst_float>(rmin), static_cast<bst_float>(rmax),
                                        static_cast<bst_float>(wmin), last_fvalue);
            ++summary_size;
            this->SetSize(summary_size);
          }
          if (summary_size == max_size) {
            next_goal = sum_total * 2.0f + 1e-5f;
          } else {
            next_goal = static_cast<bst_float>(summary_size * sum_total / max_size);
          }
        } else if (rmax >= next_goal) {
          LOG(DEBUG) << "INFO: rmax=" << rmax << ", sum_total=" << sum_total
                     << ", next_goal=" << next_goal << ", size=" << summary_size;
        }
        rmin = rmax;
        wmin = weights[c.index];
        last_fvalue = c.fvalue;
      } else {
        wmin += weights[c.index];
      }
    }

    if (col_size != 0) {
      auto summary_size = this->Size();
      double rmax = rmin + wmin;
      if (summary_size == 0 || last_fvalue > data_[summary_size - 1].value) {
        CHECK_LE(summary_size, max_size) << "Finalize: invalid maximum size, max_size=" << max_size
                                         << ", stemp.current_elements=" << summary_size;
        data_[summary_size] = Entry(static_cast<bst_float>(rmin), static_cast<bst_float>(rmax),
                                    static_cast<bst_float>(wmin), last_fvalue);
        ++summary_size;
        this->SetSize(summary_size);
      }
    }
  }
  /*!
   * \brief prune current summary in place.
   *
   * \param maxsize size we can afford in the pruned sketch
   */
  void SetPrune(size_t maxsize) {
    if (maxsize == 0) {
      this->current_elements_ = 0;
      return;
    }
    auto const src_size = this->current_elements_;
    if (src_size <= maxsize) {
      return;
    }
    // Use raw pointers in this hot loop to avoid per-access Span bounds checks.
    auto const *src_data = this->data_.data();
    auto *dst_data = data_.data();
    if (maxsize == 1) {
      dst_data[0] = src_data[0];
      this->current_elements_ = 1;
      return;
    }
    const RType begin = src_data[0].rmax;
    const RType range = src_data[src_size - 1].rmin - src_data[0].rmax;
    const size_t n = maxsize - 1;
    dst_data[0] = src_data[0];
    this->current_elements_ = 1;
    // lastidx is used to avoid duplicated records
    size_t i = 1, lastidx = 0;
    for (size_t k = 1; k < n; ++k) {
      RType dx2 = 2 * ((k * range) / n + begin);
      // find first i such that  d < (rmax[i+1] + rmin[i+1]) / 2
      while (i < src_size - 1 && dx2 >= src_data[i + 1].rmax + src_data[i + 1].rmin) {
        ++i;
      }
      if (i == src_size - 1) break;
      if (dx2 < src_data[i].RMinNext() + src_data[i + 1].RMaxPrev()) {
        if (i != lastidx) {
          dst_data[current_elements_++] = src_data[i];
          lastidx = i;
        }
      } else {
        if (i + 1 != lastidx) {
          dst_data[current_elements_++] = src_data[i + 1];
          lastidx = i + 1;
        }
      }
    }
    if (lastidx != src_size - 1) {
      dst_data[current_elements_++] = src_data[src_size - 1];
    }
  }

  /*!
   * \brief Materialize histogram cut values from this summary.
   *
   * If the summary already fits within max_bin, this reuses the exact retained values. Otherwise
   * it answers evenly spaced interior rank queries from the summary, forces the resulting cuts to
   * be strictly increasing, and appends the final sentinel upper bound required by HistogramCuts.
   */
  [[nodiscard]] std::vector<DType> QueryCutValues(std::size_t max_bin) const {
    if (this->Empty()) {
      return {static_cast<DType>(1e-5f)};
    }

    auto n_entries = this->Size();
    std::vector<DType> cut_values;
    cut_values.reserve(std::min(n_entries, max_bin) + 1);

    auto advance_to_next_distinct = [&](std::size_t cursor, DType value) {
      while (cursor < n_entries && this->data_[cursor].value <= value) {
        ++cursor;
      }
      return cursor;
    };

    auto last_cut = this->data_[0].value;
    auto next_value_cursor = advance_to_next_distinct(1, last_cut);

    if (n_entries <= max_bin) {
      while (next_value_cursor < n_entries) {
        auto cpt = this->data_[next_value_cursor].value;
        cut_values.push_back(cpt);
        last_cut = cpt;
        next_value_cursor = advance_to_next_distinct(next_value_cursor + 1, last_cut);
      }
    } else {
      auto total = static_cast<double>(this->data_[n_entries - 1].rmax);
      std::size_t query_cursor = 0;
      for (std::size_t i = 1; i < max_bin; ++i) {
        auto rank = static_cast<double>(i) * total / static_cast<double>(max_bin);
        auto rank2 = static_cast<double>(2.0) * rank;
        while (query_cursor < n_entries - 2 &&
               rank2 >= static_cast<double>(this->data_[query_cursor + 1].rmin +
                                            this->data_[query_cursor + 1].rmax)) {
          ++query_cursor;
        }
        auto const &queried = rank2 < static_cast<double>(this->data_[query_cursor].RMinNext() +
                                                          this->data_[query_cursor + 1].RMaxPrev())
                                  ? this->data_[query_cursor]
                                  : this->data_[query_cursor + 1];
        auto cpt = queried.value;
        if (cpt <= last_cut) {
          next_value_cursor = advance_to_next_distinct(next_value_cursor, last_cut);
          if (next_value_cursor == n_entries) {
            break;
          }
          cpt = this->data_[next_value_cursor].value;
        } else if (next_value_cursor < n_entries && this->data_[next_value_cursor].value <= cpt) {
          next_value_cursor = advance_to_next_distinct(next_value_cursor + 1, cpt);
        }
        cut_values.push_back(cpt);
        last_cut = cpt;
      }
    }

    auto cpt = this->data_[n_entries - 1].value;
    cut_values.push_back(cpt + (std::fabs(cpt) + static_cast<DType>(1e-5f)));
    return cut_values;
  }
  /*!
   * \brief combine `other` into `this`.
   *
   * \param other Input summary to combine with `this`.
   * \param workspace Optional entry buffer for temporary merged entries.
   */
  void SetCombine(const WQSummary &other, std::vector<Entry> *workspace = nullptr) {
    if (other.Empty()) {
      return;
    }
    if (this->data_.size() == 0) {
      this->current_elements_ = 0;
      return;
    }
    if (this->Empty()) {
      CHECK_GE(this->data_.size(), other.current_elements_);
      this->CopyFrom(other);
      return;
    }
    size_t const merged_size = this->current_elements_ + other.current_elements_;
    CHECK_GE(this->data_.size(), merged_size);

    std::vector<Entry> owned_workspace;
    if (workspace == nullptr) {
      workspace = &owned_workspace;
    }
    if (workspace->size() < merged_size) {
      workspace->resize(merged_size);
    }

    WQSummary<DType, RType> merged{Span<Entry>{workspace->data(), merged_size}, 0};
    // Merge with raw pointers to avoid Span bounds checks inside the tight loop.
    const Entry *a = this->data_.data(), *a_end = this->data_.data() + this->current_elements_;
    const Entry *b = other.data_.data(), *b_end = other.data_.data() + other.current_elements_;
    // extended rmin value
    RType aprev_rmin = 0, bprev_rmin = 0;
    Entry *dst = merged.data_.data();
    while (a != a_end && b != b_end) {
      // duplicated value entry
      if (a->value == b->value) {
        *dst = Entry(a->rmin + b->rmin, a->rmax + b->rmax, a->wmin + b->wmin, a->value);
        aprev_rmin = a->RMinNext();
        bprev_rmin = b->RMinNext();
        ++dst;
        ++a;
        ++b;
      } else if (a->value < b->value) {
        *dst = Entry(a->rmin + bprev_rmin, a->rmax + b->RMaxPrev(), a->wmin, a->value);
        aprev_rmin = a->RMinNext();
        ++dst;
        ++a;
      } else {
        *dst = Entry(b->rmin + aprev_rmin, b->rmax + a->RMaxPrev(), b->wmin, b->value);
        bprev_rmin = b->RMinNext();
        ++dst;
        ++b;
      }
    }
    if (a != a_end) {
      RType brmax = (b_end - 1)->rmax;
      do {
        *dst = Entry(a->rmin + bprev_rmin, a->rmax + brmax, a->wmin, a->value);
        ++dst;
        ++a;
      } while (a != a_end);
    }
    if (b != b_end) {
      RType armax = (a_end - 1)->rmax;
      do {
        *dst = Entry(b->rmin + aprev_rmin, b->rmax + armax, b->wmin, b->value);
        ++dst;
        ++b;
      } while (b != b_end);
    }
    merged.current_elements_ = dst - merged.data_.data();

    const RType tol = 10;
    RType err_mingap, err_maxgap, err_wgap;
    merged.FixError(&err_mingap, &err_maxgap, &err_wgap);
    if (err_mingap > tol || err_maxgap > tol || err_wgap > tol) {
      LOG(INFO) << "mingap=" << err_mingap << ", maxgap=" << err_maxgap << ", wgap=" << err_wgap;
    }
    CHECK(merged.current_elements_ <= this->current_elements_ + other.current_elements_)
        << "bug in combine";

    std::copy_n(merged.data_.data(), merged.current_elements_, this->data_.data());
    this->current_elements_ = merged.current_elements_;
  }

 protected:
  /*!
   * \brief Rebind underlying storage span while preserving current logical size.
   */
  void SetStorage(Span<Entry> storage) {
    data_ = storage;
    CHECK_LE(current_elements_, data_.size());
  }
  /*!
   * \brief Reset storage binding and clear logical size.
   */
  void ResetStorage() {
    data_ = Span<Entry>{};
    current_elements_ = 0;
  }

 private:
  /*! \brief data field */
  Span<Entry> data_;
  /*! \brief number of elements in the summary */
  size_t current_elements_;
  // try to fix rounding error
  // and re-establish invariance
  void FixError(RType *err_mingap, RType *err_maxgap, RType *err_wgap) const {
    *err_mingap = 0;
    *err_maxgap = 0;
    *err_wgap = 0;
    RType prev_rmin = 0, prev_rmax = 0;
    // Use raw pointer for the correction pass to avoid Span bounds checks.
    auto *entries = data_.data();
    for (size_t i = 0; i < this->current_elements_; ++i) {
      if (entries[i].rmin < prev_rmin) {
        entries[i].rmin = prev_rmin;
        *err_mingap = std::max(*err_mingap, prev_rmin - entries[i].rmin);
      } else {
        prev_rmin = entries[i].rmin;
      }
      if (entries[i].rmax < prev_rmax) {
        entries[i].rmax = prev_rmax;
        *err_maxgap = std::max(*err_maxgap, prev_rmax - entries[i].rmax);
      }
      RType rmin_next = entries[i].RMinNext();
      if (entries[i].rmax < rmin_next) {
        entries[i].rmax = rmin_next;
        *err_wgap = std::max(*err_wgap, entries[i].rmax - rmin_next);
      }
      prev_rmax = entries[i].rmax;
    }
  }
};

template <typename DType = bst_float, typename RType = bst_float>
struct Queue {
  using QEntry = std::pair<DType, RType>;  // value, weight

  std::vector<QEntry> queue;
  size_t max_size{1};

  explicit Queue(size_t max_size_in = 1) {
    CHECK_GE(max_size_in, 1);
    max_size = max_size_in;
    queue.reserve(1);
  }

  auto Size() const { return queue.size(); }

  // push element to the queue, return false if the queue is full and need to be flushed
  bool Push(DType x, RType w) {
    if (queue.empty() || queue.back().first != x) {
      // Keep capacity at 1 for tiny queues, reserve max capacity lazily.
      if (queue.size() == 1 && queue.capacity() == 1) {
        queue.reserve(max_size);
      }
      if (queue.size() == max_size) {
        return false;
      }
      queue.emplace_back(x, w);
      return true;
    }
    queue.back().second += w;
    return true;
  }

  template <typename Summary>
  void PopSummary(Summary *out) {
    CHECK(out);
    out->Reserve(queue.size());
    std::sort(queue.begin(), queue.end(),
              [](QEntry const &l, QEntry const &r) { return l.first < r.first; });
    out->SetFromSorted(queue);
    queue.clear();
  }
};

struct WQSummaryContainer : public WQSummary<> {
  std::vector<WQSummary<>::Entry> space;
  WQSummaryContainer() : WQSummary<>(Span<WQSummary<>::Entry>{}, 0) {}

  WQSummaryContainer(WQSummaryContainer const &src) = delete;

  WQSummaryContainer(WQSummaryContainer &&src) noexcept
      : WQSummary<>(Span<WQSummary<>::Entry>{}, 0), space{std::move(src.space)} {
    this->SetStorage({dmlc::BeginPtr(this->space), this->space.size()});
    this->SetSize(src.Size());
    src.ResetStorage();
  }

  WQSummaryContainer &operator=(WQSummaryContainer const &src) = delete;

  WQSummaryContainer &operator=(WQSummaryContainer &&src) noexcept {
    if (this == &src) {
      return *this;
    }
    this->space = std::move(src.space);
    this->SetStorage({dmlc::BeginPtr(this->space), this->space.size()});
    this->SetSize(src.Size());
    src.ResetStorage();
    return *this;
  }

  void Reserve(size_t size) {
    if (size > space.size()) {
      space.resize(size);
    }
    this->SetStorage({dmlc::BeginPtr(space), space.size()});
  }
};

/*! \brief Weighted quantile sketch algorithm using merge/prune. */
class WQuantileSketch {
 public:
  // Safety factor used to oversample the internal sketch relative to the target rank
  // resolution. User-facing epsilon remains the target rank guarantee; `kFactor`
  // only affects how much summary storage we reserve to achieve it.
  static float constexpr kFactor = 2.0;

 public:
  using Summary = WQSummary<>;
  using Entry = typename WQSummary<>::Entry;
  using SummaryContainer = WQSummaryContainer;
  WQuantileSketch() = default;
  WQuantileSketch(size_t maxn, double eps) {
    limit_size_ = LimitSizeLevel(maxn, eps);
    inqueue_ = Queue<>(limit_size_ * 2);
    data_.clear();
    level_.clear();
  }

  [[nodiscard]] size_t NumElements() const { return num_elements_; }

  static size_t LimitSizeLevel(size_t maxn, double eps) {
    if (maxn == 0) {
      // Empty columns can appear in distributed column-split settings.
      return 1;
    }
    auto const internal_eps = eps / kFactor;
    size_t nlevel = 1;
    size_t limit_size = 1;
    while (true) {
      limit_size = static_cast<size_t>(ceil(nlevel / internal_eps)) + 1;
      limit_size = std::min(maxn, limit_size);
      size_t n = (1ULL << nlevel);
      if (n * limit_size >= maxn) break;
      ++nlevel;
    }
    // check invariant
    size_t n = (1ULL << nlevel);
    CHECK(n * limit_size >= maxn) << "invalid init parameter";
    CHECK(nlevel <=
          std::max(static_cast<size_t>(1), static_cast<size_t>(limit_size * internal_eps)))
        << "invalid init parameter";
    return limit_size;
  }

  /*!
   * \brief add an element to a sketch
   * \param x The element added to the sketch
   * \param w The weight of the element.
   */
  void Push(bst_float x, bst_float w = 1) {
    if (w == static_cast<bst_float>(0)) return;
    ++num_elements_;
    if (!inqueue_.Push(x, w)) {
      inqueue_.PopSummary(&temp_);
      this->PushSummary(&temp_);
      inqueue_.Push(x, w);
    }
  }

  /*!
   * \brief Add sorted column entries into this sketch.
   *
   * \param column Sorted column entries in ascending order by feature value.
   * \param weights Row weights.
   * \param num_retained_items Target number of summary items to retain from sorted input.
   */
  void PushSorted(common::Span<::xgboost::Entry const> column, std::vector<float> const &weights,
                  size_t num_retained_items) {
    CHECK_GE(num_retained_items, 1);
    if (weights.empty()) {
      num_elements_ += column.size();
    } else {
      num_elements_ +=
          std::count_if(column.cbegin(), column.cend(), [&](::xgboost::Entry const &entry) {
            return weights[entry.index] != static_cast<float>(0);
          });
    }
    auto const max_size = num_retained_items;
    this->temp_.Reserve(max_size + 1);
    this->temp_.SetPruneSorted(column, weights, max_size);
    if (!column.empty()) {
      this->PushSummary(&temp_);
    }
  }

  /*! \brief push up a prepared summary */
  void PushSummary(WQSummaryContainer *summary) {
    CHECK(summary);
    summary->Reserve(limit_size_ * 2);
    size_t l = 0;
    // Level-wise merge/prune with carry propagation.
    //
    // Reference:
    //   Greenwald, M. and Khanna, S. "Space-efficient Online Computation of
    //   Quantile Summaries", SIGMOD 2001.
    while (true) {
      this->LazyInitLevel(l + 1);
      // Clamp the incoming summary to per-level capacity before combining.
      summary->SetPrune(limit_size_);
      // Merge with the resident level summary.
      summary->SetCombine(level_[l], &combine_workspace_);
      // Level[l] is consumed into `summary`. Clear it before carry propagation.
      level_[l].Clear();
      // If merged summary fits, store at this level. Otherwise carry upward.
      if (summary->Size() <= limit_size_) {
        break;
      }
      ++l;
    }

    // First level where merged summary fits.
    level_[l].CopyFrom(*summary);
  }

 public:
  /*! \brief get the summary after finalize */
  [[nodiscard]] WQSummaryContainer GetSummary(size_t max_size) {
    // Flush pending queue into level summaries first.
    inqueue_.PopSummary(&temp_);
    this->PushSummary(&temp_);

    auto const prune_size = std::max(max_size, limit_size_);
    // Reserve based on observed live storage after local merge.
    // This keeps memory use small when the sketch has very few entries (e.g. sparse
    // columns / few local instances) while still reserving enough for immediate merges.
    std::size_t observed_level_entries = 0;
    for (auto const &level_summary : level_) {
      observed_level_entries += level_summary.Size();
    }
    auto initial_reserve = std::min<std::size_t>(observed_level_entries, prune_size + limit_size_);
    WQSummaryContainer out;
    if (initial_reserve > 0) {
      out.Reserve(initial_reserve);
    }

    // Merge all levels into out.
    for (auto &level_summary : level_) {
      auto combine_needed = out.Size() + level_summary.Size();
      if (combine_needed > out.space.size()) {
        out.Reserve(combine_needed);
      }
      out.SetCombine(level_summary, &combine_workspace_);
      out.SetPrune(prune_size);
    }
    out.SetPrune(max_size);
    return out;
  }

 private:
  // initialize level space to at least nlevel
  void LazyInitLevel(size_t nlevel) {
    if (level_.size() >= nlevel) return;
    data_.resize(limit_size_ * nlevel);
    level_.clear();
    level_.reserve(nlevel);
    for (size_t l = 0; l < nlevel; ++l) {
      level_.emplace_back(Span<Entry>{data_.data() + l * limit_size_, limit_size_}, 0);
    }
  }
  // input data queue
  Queue<> inqueue_{1};
  // size of summary in each level
  size_t limit_size_{1};
  // the level of each summaries
  std::vector<WQSummary<>> level_;
  // content of the summary
  std::vector<WQSummary<>::Entry> data_;
  // temporal summary, used for temp-merge
  WQSummaryContainer temp_;
  // reusable workspace for combine-prune operations
  std::vector<Entry> combine_workspace_;
  // Number of source elements represented by this sketch.
  size_t num_elements_{0};
};

[[nodiscard]] inline double SketchEpsilon(bst_bin_t max_bins, std::size_t num_samples) {
  auto const n = std::max<std::size_t>(1, num_samples);
  auto const n_bins = std::min<std::size_t>(static_cast<std::size_t>(max_bins), n);
  return 1.0 / static_cast<double>(n_bins);
}

// Per-feature summary size for a sketch that represents `num_samples`. `num_samples`
// can be an exact per-feature count or a conservative approximation when a tighter count
// is not available on the current path.
[[nodiscard]] inline std::size_t SketchSummaryBudget(bst_bin_t max_bins, std::size_t num_samples) {
  return WQuantileSketch::LimitSizeLevel(num_samples, SketchEpsilon(max_bins, num_samples));
}

namespace detail {
inline std::vector<float> UnrollGroupWeights(MetaInfo const &info) {
  auto const &group_weights = info.weights_.HostVector();
  if (group_weights.empty()) {
    return group_weights;
  }

  auto const &group_ptr = info.group_ptr_;
  CHECK_GE(group_ptr.size(), 2);
  CHECK_EQ(group_weights.size(), group_ptr.size() - 1) << error::GroupWeight();
  CHECK_EQ(group_ptr.back(), info.num_row_)
      << error::GroupSize() << " the number of rows from the data.";

  std::vector<float> out(info.num_row_);
  size_t cur_group = 0;
  for (bst_idx_t i = 0; i < info.num_row_; ++i) {
    while (cur_group + 1 < group_ptr.size() && i >= group_ptr[cur_group + 1]) {
      ++cur_group;
    }
    out[i] = group_weights[cur_group];
  }
  return out;
}
}  // namespace detail

class HistogramCuts;

template <typename Batch, typename IsValid>
std::vector<bst_idx_t> CalcColumnSize(Batch const &batch, bst_feature_t const n_columns,
                                      size_t const n_threads, IsValid &&is_valid) {
  std::vector<std::vector<bst_idx_t>> column_sizes_tloc(n_threads);
  for (auto &column : column_sizes_tloc) {
    column.resize(n_columns, 0);
  }

  ParallelFor(batch.Size(), n_threads, [&](omp_ulong i) {
    auto &local_column_sizes = column_sizes_tloc.at(omp_get_thread_num());
    auto const &line = batch.GetLine(i);
    for (size_t j = 0; j < line.Size(); ++j) {
      auto elem = line.GetElement(j);
      if (is_valid(elem)) {
        local_column_sizes[elem.column_idx]++;
      }
    }
  });
  // reduce to first thread
  auto &entries_per_columns = column_sizes_tloc.front();
  CHECK_EQ(entries_per_columns.size(), static_cast<size_t>(n_columns));
  for (size_t i = 1; i < n_threads; ++i) {
    CHECK_EQ(column_sizes_tloc[i].size(), static_cast<size_t>(n_columns));
    for (size_t j = 0; j < n_columns; ++j) {
      entries_per_columns[j] += column_sizes_tloc[i][j];
    }
  }
  return entries_per_columns;
}

template <typename Batch, typename IsValid>
std::vector<bst_feature_t> LoadBalance(Batch const &batch, size_t nnz, bst_feature_t n_columns,
                                       size_t const nthreads, IsValid &&is_valid) {
  /* Some sparse datasets have their mass concentrating on small number of features.  To
   * avoid waiting for a few threads running forever, we here distribute different number
   * of columns to different threads according to number of entries.
   */
  size_t const total_entries = nnz;
  size_t const entries_per_thread = DivRoundUp(total_entries, nthreads);

  // Need to calculate the size for each batch.
  std::vector<bst_idx_t> entries_per_columns = CalcColumnSize(batch, n_columns, nthreads, is_valid);
  std::vector<bst_feature_t> cols_ptr(nthreads + 1, 0);
  size_t count{0};
  size_t current_thread{1};

  for (auto col : entries_per_columns) {
    cols_ptr.at(current_thread)++;  // add one column to thread
    count += col;
    CHECK_LE(count, total_entries);
    if (count > entries_per_thread) {
      current_thread++;
      count = 0;
      cols_ptr.at(current_thread) = cols_ptr[current_thread - 1];
    }
  }
  // Idle threads.
  for (; current_thread < cols_ptr.size() - 1; ++current_thread) {
    cols_ptr[current_thread + 1] = cols_ptr[current_thread];
  }
  return cols_ptr;
}

/*!
 * A sketch matrix storing sketches for each feature.
 */
class HostSketchContainer {
 protected:
  using WQSketch = WQuantileSketch;
  std::vector<WQSketch> sketches_;
  std::vector<std::set<float>> categories_;
  std::vector<FeatureType> const feature_types_;

  std::vector<bst_idx_t> columns_size_;
  bst_bin_t max_bins_;
  bool use_group_ind_{false};
  int32_t n_threads_;
  bool has_categorical_{false};
  Monitor monitor_;

 public:
  /* \brief Initialize necessary info.
   *
   * \param columns_size Size of each column.
   * \param max_bin maximum number of bins for each feature.
   * \param use_group whether is assigned to group to data instance.
   */
  HostSketchContainer(Context const *ctx, bst_bin_t max_bin,
                      common::Span<FeatureType const> feature_types,
                      std::vector<bst_idx_t> columns_size, bool use_group);

  static bool UseGroup(MetaInfo const &info) {
    size_t const num_groups = info.group_ptr_.size() == 0 ? 0 : info.group_ptr_.size() - 1;
    // Use group index for weights?
    bool const use_group_ind = num_groups != 0 && (info.weights_.Size() != info.num_row_);
    return use_group_ind;
  }

  /* \brief Push a CSR matrix. */
  void PushRowPage(SparsePage const &page, MetaInfo const &info, Span<float const> hessian = {});

  template <typename Batch>
  void PushAdapterBatch(Batch const &batch, size_t base_rowid, MetaInfo const &info, float missing);

  /**
   * \brief Push a sorted CSC page.
   */
  void PushColPage(SparsePage const &page, MetaInfo const &info, Span<float const> hessian);

  [[nodiscard]] HistogramCuts MakeCuts(Context const *ctx, MetaInfo const &info);

 protected:
  template <typename Batch, typename IsValid>
  void PushRowPageImpl(Batch const &batch, std::size_t base_rowid, OptionalWeights weights,
                       size_t nnz, size_t n_features, bool is_dense, IsValid is_valid) {
    auto thread_columns_ptr = LoadBalance(batch, nnz, n_features, n_threads_, is_valid);
    ParallelFor(static_cast<std::size_t>(n_threads_), n_threads_, [&](std::size_t tid) {
      auto const begin = thread_columns_ptr[tid];
      auto const end = thread_columns_ptr[tid + 1];

      // do not iterate if no columns are assigned to the thread
      if (begin < end && end <= n_features) {
        for (size_t ridx = 0; ridx < batch.Size(); ++ridx) {
          auto const &line = batch.GetLine(ridx);
          auto w = weights[ridx + base_rowid];
          if (is_dense) {
            for (size_t ii = begin; ii < end; ii++) {
              auto elem = line.GetElement(ii);
              if (is_valid(elem)) {
                if (IsCat(feature_types_, ii)) {
                  categories_[ii].emplace(elem.value);
                } else {
                  sketches_[ii].Push(elem.value, w);
                }
              }
            }
          } else {
            for (size_t i = 0; i < line.Size(); ++i) {
              auto const &elem = line.GetElement(i);
              if (is_valid(elem) && elem.column_idx >= begin && elem.column_idx < end) {
                if (IsCat(feature_types_, elem.column_idx)) {
                  categories_[elem.column_idx].emplace(elem.value);
                } else {
                  sketches_[elem.column_idx].Push(elem.value, w);
                }
              }
            }
          }
        }
      }
    });
  }

 private:
  // Merge categorical values from all workers.
  [[nodiscard]] auto AllreduceCategories(Context const *ctx, MetaInfo const &info,
                                         common::Span<bst_feature_t const> categorical_features)
      -> std::vector<std::set<float>>;

  // Merge numeric sketches from all workers.
  [[nodiscard]] auto AllReduce(Context const *ctx, MetaInfo const &info,
                               common::Span<bst_feature_t const> numeric_features)
      -> std::vector<WQSketch::SummaryContainer>;
};
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_QUANTILE_H_
