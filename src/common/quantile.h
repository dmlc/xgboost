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
#include <cstring>
#include <iostream>
#include <set>
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
    /*!
     * \brief debug function,  check Valid
     * \param eps the tolerate level for violating the relation
     */
    inline void CheckValid(RType eps = 0) const {
      CHECK(rmin >= 0 && rmax >= 0 && wmin >= 0) << "nonneg constraint";
      CHECK(rmax - rmin - wmin > -eps) << "relation constraint: min/max";
    }
    /*! \return rmin estimation for v strictly bigger than value */
    XGBOOST_DEVICE inline RType RMinNext() const { return rmin + wmin; }
    /*! \return rmax estimation for v strictly smaller than value */
    XGBOOST_DEVICE inline RType RMaxPrev() const { return rmax - wmin; }

    friend std::ostream &operator<<(std::ostream &os, Entry const &e) {
      os << "rmin: " << e.rmin << ", "
         << "rmax: " << e.rmax << ", "
         << "wmin: " << e.wmin << ", "
         << "value: " << e.value;
      return os;
    }
  };
  /*! \brief summary entries */
  Span<Entry> data;
  // constructor
  WQSummary(Entry *data, size_t size) : data{data, size} {}
  /*!
   * \return the maximum error of the Summary
   */
  inline RType MaxError() const {
    RType res = data[0].rmax - data[0].rmin - data[0].wmin;
    for (size_t i = 1; i < data.size(); ++i) {
      res = std::max(data[i].RMaxPrev() - data[i - 1].RMinNext(), res);
      res = std::max(data[i].rmax - data[i].rmin - data[i].wmin, res);
    }
    return res;
  }
  /*!
   * \brief query qvalue, start from istart
   * \param qvalue the value we query for
   * \param istart starting position
   */
  inline Entry Query(DType qvalue, size_t &istart) const {  // NOLINT(*)
    while (istart < data.size() && qvalue > data[istart].value) {
      ++istart;
    }
    if (istart == data.size()) {
      RType rmax = data.back().rmax;
      return Entry(rmax, rmax, 0.0f, qvalue);
    }
    if (qvalue == data[istart].value) {
      return data[istart];
    } else {
      if (istart == 0) {
        return Entry(0.0f, 0.0f, 0.0f, qvalue);
      } else {
        return Entry(data[istart - 1].RMinNext(), data[istart].RMaxPrev(), 0.0f, qvalue);
      }
    }
  }
  /*! \return maximum rank in the summary */
  inline RType MaxRank() const { return data.back().rmax; }
  /*!
   * \brief copy content from src
   * \param src source sketch
   */
  inline void CopyFrom(const WQSummary &src) {
    if (!src.data.data()) {
      CHECK_EQ(src.data.size(), 0);
      data = Span<Entry>{data.data(), std::size_t{0}};
      return;
    }
    if (!data.data()) {
      CHECK_EQ(this->data.size(), 0);
      CHECK_EQ(src.data.size(), 0);
      return;
    }
    std::memcpy(data.data(), src.data.data(), sizeof(Entry) * src.data.size());
    data = Span<Entry>{data.data(), src.data.size()};
  }
  inline void MakeFromSorted(const Entry *entries, size_t n) {
    auto out = data.data();
    size_t size{0};
    for (size_t i = 0; i < n;) {
      size_t j = i + 1;
      // ignore repeated values
      for (; j < n && entries[j].value == entries[i].value; ++j) {
      }
      out[size++] = Entry(entries[i].rmin, entries[i].rmax, entries[i].wmin, entries[i].value);
      i = j;
    }
    data = Span<Entry>{out, size};
  }
  /*!
   * \brief debug function, validate whether the summary
   *  run consistency check to check if it is a valid summary
   * \param eps the tolerate error level, used when RType is floating point and
   *        some inconsistency could occur due to rounding error
   */
  inline void CheckValid(RType eps) const {
    for (size_t i = 0; i < data.size(); ++i) {
      data[i].CheckValid(eps);
      if (i != 0) {
        CHECK(data[i].rmin >= data[i - 1].rmin + data[i - 1].wmin) << "rmin range constraint";
        CHECK(data[i].rmax >= data[i - 1].rmax + data[i].wmin) << "rmax range constraint";
      }
    }
  }

  /*!
   * \brief set current summary to be pruned summary of src
   *        assume data field is already allocated to be at least maxsize
   * \param src source summary
   * \param maxsize size we can afford in the pruned sketch
   */
  void SetPrune(const WQSummary &src, size_t maxsize) {
    if (src.data.size() <= maxsize) {
      this->CopyFrom(src);
      return;
    }
    auto out = data.data();
    const RType begin = src.data[0].rmax;
    const RType range = src.data.back().rmin - src.data[0].rmax;
    const size_t n = maxsize - 1;
    out[0] = src.data[0];
    size_t size{1};
    // lastidx is used to avoid duplicated records
    size_t i = 1, lastidx = 0;
    for (size_t k = 1; k < n; ++k) {
      RType dx2 = 2 * ((k * range) / n + begin);
      // find first i such that  d < (rmax[i+1] + rmin[i+1]) / 2
      while (i < src.data.size() - 1 && dx2 >= src.data[i + 1].rmax + src.data[i + 1].rmin) ++i;
      if (i == src.data.size() - 1) break;
      if (dx2 < src.data[i].RMinNext() + src.data[i + 1].RMaxPrev()) {
        if (i != lastidx) {
          out[size++] = src.data[i];
          lastidx = i;
        }
      } else {
        if (i + 1 != lastidx) {
          out[size++] = src.data[i + 1];
          lastidx = i + 1;
        }
      }
    }
    if (lastidx != src.data.size() - 1) {
      out[size++] = src.data.back();
    }
    data = Span<Entry>{out, size};
  }
  /*!
   * \brief set current summary to be merged summary of sa and sb
   * \param sa first input summary to be merged
   * \param sb second input summary to be merged
   */
  inline void SetCombine(const WQSummary &sa, const WQSummary &sb) {
    if (sa.data.size() == 0) {
      this->CopyFrom(sb);
      return;
    }
    if (sb.data.size() == 0) {
      this->CopyFrom(sa);
      return;
    }
    CHECK(sa.data.size() > 0 && sb.data.size() > 0);
    const Entry *a = sa.data.data(), *a_end = sa.data.data() + sa.data.size();
    const Entry *b = sb.data.data(), *b_end = sb.data.data() + sb.data.size();
    // extended rmin value
    RType aprev_rmin = 0, bprev_rmin = 0;
    Entry *dst = this->data.data();
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
    this->data = Span<Entry>{this->data.data(), static_cast<size_t>(dst - this->data.data())};
    const RType tol = 10;
    RType err_mingap, err_maxgap, err_wgap;
    this->FixError(&err_mingap, &err_maxgap, &err_wgap);
    if (err_mingap > tol || err_maxgap > tol || err_wgap > tol) {
      LOG(INFO) << "mingap=" << err_mingap << ", maxgap=" << err_maxgap << ", wgap=" << err_wgap;
    }
    CHECK(data.size() <= sa.data.size() + sb.data.size()) << "bug in combine";
  }
  // helper function to print the current content of sketch
  inline void Print() const {
    for (size_t i = 0; i < this->data.size(); ++i) {
      LOG(CONSOLE) << "[" << i << "] rmin=" << data[i].rmin << ", rmax=" << data[i].rmax
                   << ", wmin=" << data[i].wmin << ", v=" << data[i].value;
    }
  }
  // try to fix rounding error
  // and re-establish invariance
  inline void FixError(RType *err_mingap, RType *err_maxgap, RType *err_wgap) const {
    *err_mingap = 0;
    *err_maxgap = 0;
    *err_wgap = 0;
    RType prev_rmin = 0, prev_rmax = 0;
    for (size_t i = 0; i < this->data.size(); ++i) {
      if (data[i].rmin < prev_rmin) {
        data[i].rmin = prev_rmin;
        *err_mingap = std::max(*err_mingap, prev_rmin - data[i].rmin);
      } else {
        prev_rmin = data[i].rmin;
      }
      if (data[i].rmax < prev_rmax) {
        data[i].rmax = prev_rmax;
        *err_maxgap = std::max(*err_maxgap, prev_rmax - data[i].rmax);
      }
      RType rmin_next = data[i].RMinNext();
      if (data[i].rmax < rmin_next) {
        data[i].rmax = rmin_next;
        *err_wgap = std::max(*err_wgap, data[i].rmax - rmin_next);
      }
      prev_rmax = data[i].rmax;
    }
  }
};

template <typename DType = bst_float, typename RType = bst_float>
struct Queue {
  struct QEntry {
    DType value;
    RType weight;
    QEntry() = default;
    QEntry(DType value, RType weight) : value(value), weight(weight) {}
    inline bool operator<(QEntry const &b) const { return value < b.value; }
  };

  std::vector<QEntry> queue;
  size_t qtail{0};
  size_t max_size{1};

  explicit Queue(size_t max_size_in = 1) {
    CHECK_GE(max_size_in, 1);
    max_size = max_size_in;
    queue.resize(1);
    qtail = 0;
  }

  // push element to the queue, return false if the queue is full and need to be flushed
  inline bool Push(DType x, RType w) {
    if (qtail == 0 || queue[qtail - 1].value != x) {
      if (qtail == queue.size() && queue.size() == 1) {
        queue.resize(max_size);
      }
      if (qtail == queue.size()) {
        return false;
      }
      queue[qtail++] = QEntry(x, w);
      return true;
    }
    queue[qtail - 1].weight += w;
    return true;
  }

  inline void PopSummary(WQSummary<DType, RType> *out) {
    std::sort(queue.begin(), queue.begin() + qtail);
    auto out_data = out->data.data();
    size_t out_size = 0;
    RType wsum = 0;
    for (size_t i = 0; i < qtail;) {
      size_t j = i + 1;
      RType w = queue[i].weight;
      while (j < qtail && queue[j].value == queue[i].value) {
        w += queue[j].weight;
        ++j;
      }
      out_data[out_size++] =
          typename WQSummary<DType, RType>::Entry(wsum, wsum + w, w, queue[i].value);
      wsum += w;
      i = j;
    }
    out->data = Span<typename WQSummary<DType, RType>::Entry>{out_data, out_size};
    qtail = 0;
  }
};

struct WQSummaryContainer : public WQSummary<> {
  std::vector<WQSummary<>::Entry> space;
  WQSummaryContainer(WQSummaryContainer const &src) : WQSummary<>(nullptr, 0) {
    this->space = src.space;
    this->data = Span<Entry>{dmlc::BeginPtr(this->space), src.data.size()};
  }
  WQSummaryContainer() : WQSummary<>(nullptr, 0) {}
  inline void Reserve(size_t size) {
    auto current_size = this->data.size();
    if (size > space.size()) {
      space.resize(size);
    }
    this->data = Span<Entry>{dmlc::BeginPtr(space), current_size};
  }
  inline void Reduce(WQSummary<> const &src, size_t max_nbyte) {
    this->Reserve((max_nbyte - sizeof(std::size_t)) / sizeof(WQSummary<>::Entry));
    WQSummaryContainer temp;
    temp.Reserve(this->data.size() + src.data.size());
    temp.SetCombine(*this, src);
    this->SetPrune(temp, space.size());
  }
  inline static size_t CalcMemCost(size_t nentry) {
    return sizeof(size_t) + sizeof(WQSummary<>::Entry) * nentry;
  }
  template <typename TStream>
  inline void Save(TStream &fo) const {  // NOLINT(*)
    auto size = this->data.size();
    fo.Write(&size, sizeof(size));
    if (size != 0) {
      fo.Write(this->data.data(), size * sizeof(Entry));
    }
  }
  template <typename TStream>
  inline void Load(TStream &fi) {  // NOLINT(*)
    std::size_t size{0};
    CHECK_EQ(fi.Read(&size, sizeof(size)), sizeof(size));
    this->Reserve(size);
    if (size != 0) {
      CHECK_EQ(fi.Read(this->data.data(), size * sizeof(Entry)), size * sizeof(Entry));
    }
    this->data = Span<Entry>{this->data.data(), size};
  }
};

/*! \brief Weighted quantile sketch algorithm using merge/prune. */
class WQuantileSketch {
 public:
  static float constexpr kFactor = 8.0;

 public:
  using Summary = WQSummary<>;
  using Entry = typename WQSummary<>::Entry;
  using SummaryContainer = WQSummaryContainer;

  /*!
   * \brief initialize the quantile sketch, given the performance specification
   * \param maxn maximum number of data points can be feed into sketch
   * \param eps accuracy level of summary
   */
  inline void Init(size_t maxn, double eps) {
    if (maxn == 0) {
      // Empty columns can appear in distributed column-split settings.
      // Keep internals in a valid state while preserving an empty summary.
      nlevel = 1;
      limit_size = 1;
      inqueue = Queue<>(1);
      data.clear();
      level.clear();
      return;
    }
    LimitSizeLevel(maxn, eps, &nlevel, &limit_size);
    inqueue = Queue<>(limit_size * 2);
    data.clear();
    level.clear();
  }

  inline static void LimitSizeLevel(size_t maxn, double eps, size_t *out_nlevel,
                                    size_t *out_limit_size) {
    size_t &nlevel = *out_nlevel;
    size_t &limit_size = *out_limit_size;
    nlevel = 1;
    while (true) {
      limit_size = static_cast<size_t>(ceil(nlevel / eps)) + 1;
      limit_size = std::min(maxn, limit_size);
      size_t n = (1ULL << nlevel);
      if (n * limit_size >= maxn) break;
      ++nlevel;
    }
    // check invariant
    size_t n = (1ULL << nlevel);
    CHECK(n * limit_size >= maxn) << "invalid init parameter";
    CHECK(nlevel <= std::max(static_cast<size_t>(1), static_cast<size_t>(limit_size * eps)))
        << "invalid init parameter";
  }

  /*!
   * \brief add an element to a sketch
   * \param x The element added to the sketch
   * \param w The weight of the element.
   */
  inline void Push(bst_float x, bst_float w = 1) {
    if (w == static_cast<bst_float>(0)) return;
    if (!inqueue.Push(x, w)) {
      temp.Reserve(limit_size * 2);
      inqueue.PopSummary(&temp);
      this->PushTemp();
      inqueue.Push(x, w);
    }
  }

  inline void PushSummary(WQSummary<> const &summary) {
    temp.Reserve(limit_size * 2);
    temp.SetPrune(summary, limit_size * 2);
    PushTemp();
  }

  /*! \brief push up temp */
  inline void PushTemp() {
    temp.Reserve(limit_size * 2);
    for (size_t l = 1; true; ++l) {
      this->InitLevel(l + 1);
      // check if level l is empty
      if (level[l].data.size() == 0) {
        level[l].SetPrune(temp, limit_size);
        break;
      } else {
        // level 0 is actually temp space
        level[0].SetPrune(temp, limit_size);
        temp.SetCombine(level[0], level[l]);
        if (temp.data.size() > limit_size) {
          // try next level
          level[l].data = Span<Entry>{level[l].data.data(), std::size_t{0}};
        } else {
          // if merged record is still smaller, no need to send to next level
          level[l].CopyFrom(temp);
          break;
        }
      }
    }
  }
  /*! \brief get the summary after finalize */
  inline void GetSummary(WQSummaryContainer *out) {
    if (level.size() != 0) {
      out->Reserve(limit_size * 2);
    } else {
      out->Reserve(inqueue.queue.size());
    }
    inqueue.PopSummary(out);
    if (level.size() != 0) {
      level[0].SetPrune(*out, limit_size);
      for (size_t l = 1; l < level.size(); ++l) {
        if (level[l].data.size() == 0) continue;
        if (level[0].data.size() == 0) {
          level[0].CopyFrom(level[l]);
        } else {
          out->SetCombine(level[0], level[l]);
          level[0].SetPrune(*out, limit_size);
        }
      }
      out->CopyFrom(level[0]);
    } else {
      if (out->data.size() > limit_size) {
        temp.Reserve(limit_size);
        temp.SetPrune(*out, limit_size);
        out->CopyFrom(temp);
      }
    }
  }
  // used for debug, check if the sketch is valid
  inline void CheckValid(bst_float eps) const {
    for (size_t l = 1; l < level.size(); ++l) {
      level[l].CheckValid(eps);
    }
  }
  // initialize level space to at least nlevel
  inline void InitLevel(size_t nlevel) {
    if (level.size() >= nlevel) return;
    data.resize(limit_size * nlevel);
    level.resize(nlevel, WQSummary<>(nullptr, 0));
    for (size_t l = 0; l < level.size(); ++l) {
      level[l].data = Span<Entry>{dmlc::BeginPtr(data) + l * limit_size, std::size_t{0}};
    }
  }
  // input data queue
  Queue<> inqueue;
  // number of levels
  size_t nlevel;
  // size of summary in each level
  size_t limit_size;
  // the level of each summaries
  std::vector<WQSummary<>> level;
  // content of the summary
  std::vector<WQSummary<>::Entry> data;
  // temporal summary, used for temp-merge
  WQSummaryContainer temp;
};

namespace detail {
inline std::vector<float> UnrollGroupWeights(MetaInfo const &info) {
  std::vector<float> const &group_weights = info.weights_.HostVector();
  if (group_weights.empty()) {
    return group_weights;
  }

  auto const &group_ptr = info.group_ptr_;
  CHECK_GE(group_ptr.size(), 2);

  auto n_groups = group_ptr.size() - 1;
  CHECK_EQ(info.weights_.Size(), n_groups) << error::GroupWeight();

  bst_idx_t n_samples = info.num_row_;
  std::vector<float> results(n_samples);
  CHECK_EQ(group_ptr.back(), n_samples)
      << error::GroupSize() << " the number of rows from the data.";
  size_t cur_group = 0;
  for (bst_idx_t i = 0; i < n_samples; ++i) {
    results[i] = group_weights[cur_group];
    if (i == group_ptr[cur_group + 1]) {
      cur_group++;
    }
  }
  return results;
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
class SketchContainerImpl {
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
  SketchContainerImpl(Context const *ctx, std::vector<bst_idx_t> columns_size, bst_bin_t max_bin,
                      common::Span<FeatureType const> feature_types, bool use_group);

  static bool UseGroup(MetaInfo const &info) {
    size_t const num_groups = info.group_ptr_.size() == 0 ? 0 : info.group_ptr_.size() - 1;
    // Use group index for weights?
    bool const use_group_ind = num_groups != 0 && (info.weights_.Size() != info.num_row_);
    return use_group_ind;
  }

  static uint32_t SearchGroupIndFromRow(std::vector<bst_uint> const &group_ptr,
                                        size_t const base_rowid) {
    CHECK_LT(base_rowid, group_ptr.back())
        << "Row: " << base_rowid << " is not found in any group.";
    bst_group_t group_ind = std::upper_bound(group_ptr.cbegin(), group_ptr.cend() - 1, base_rowid) -
                            group_ptr.cbegin() - 1;
    return group_ind;
  }
  // Gather sketches from all workers.
  void GatherSketchInfo(Context const *ctx, MetaInfo const &info,
                        std::vector<WQSketch::SummaryContainer> const &reduced,
                        std::vector<bst_idx_t> *p_worker_segments,
                        std::vector<bst_idx_t> *p_sketches_scan,
                        std::vector<WQSketch::Entry> *p_global_sketches);
  // Merge sketches from all workers.
  void AllReduce(Context const *ctx, MetaInfo const &info,
                 std::vector<WQSketch::SummaryContainer> *p_reduced,
                 std::vector<int32_t> *p_num_cuts);

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

  /* \brief Push a CSR matrix. */
  void PushRowPage(SparsePage const &page, MetaInfo const &info, Span<float const> hessian = {});

  void MakeCuts(Context const *ctx, MetaInfo const &info, HistogramCuts *cuts);

 private:
  // Merge all categories from other workers.
  void AllreduceCategories(Context const *ctx, MetaInfo const &info);
};

class HostSketchContainer : public SketchContainerImpl {
 public:
  using WQSketch = WQuantileSketch;

 public:
  HostSketchContainer(Context const *ctx, bst_bin_t max_bins, common::Span<FeatureType const> ft,
                      std::vector<bst_idx_t> columns_size, bool use_group);

  template <typename Batch>
  void PushAdapterBatch(Batch const &batch, size_t base_rowid, MetaInfo const &info, float missing);
};

/**
 * \brief Quantile structure accepts sorted data, extracted from histmaker.
 */
struct SortedQuantile {
  using Entry = common::WQuantileSketch::Entry;
  /*! \brief total sum of amount to be met */
  double sum_total{0.0};
  /*! \brief statistics used in the sketch */
  double rmin, wmin;
  /*! \brief last seen feature value */
  bst_float last_fvalue;
  /*! \brief current size of sketch */
  double next_goal;
  // pointer to the sketch to put things in
  common::WQuantileSketch *sketch;

  explicit SortedQuantile(common::WQuantileSketch *sketch, unsigned max_size) : sketch{sketch} {
    next_goal = -1.0f;
    rmin = wmin = 0.0f;
    sketch->temp.Reserve(max_size + 1);
    sketch->temp.data = Span<Entry>{sketch->temp.data.data(), std::size_t{0}};
  }
  /*!
   * \brief push a new element to sketch
   * \param fvalue feature value, comes in sorted ascending order
   * \param w weight
   * \param max_size
   */
  inline void Push(bst_float fvalue, bst_float w, unsigned max_size) {
    if (next_goal == -1.0f) {
      next_goal = 0.0f;
      last_fvalue = fvalue;
      wmin = w;
      return;
    }
    if (last_fvalue != fvalue) {
      double rmax = rmin + wmin;
      if (rmax >= next_goal && sketch->temp.data.size() != max_size) {
        auto &temp = sketch->temp.data;
        auto temp_data = temp.data();
        auto temp_size = temp.size();
        if (temp.empty() || last_fvalue > temp.back().value) {
          // push to sketch
          temp_data[temp_size] = common::WQuantileSketch::Entry(
              static_cast<bst_float>(rmin), static_cast<bst_float>(rmax),
              static_cast<bst_float>(wmin), last_fvalue);
          CHECK_LT(temp_size, max_size)
              << "invalid maximum size max_size=" << max_size << ", stemp.size" << temp_size;
          sketch->temp.data = Span<Entry>{temp_data, temp_size + 1};
        }
        if (sketch->temp.data.size() == max_size) {
          next_goal = sum_total * 2.0f + 1e-5f;
        } else {
          next_goal = static_cast<bst_float>(sketch->temp.data.size() * sum_total / max_size);
        }
      } else {
        if (rmax >= next_goal) {
          LOG(DEBUG) << "INFO: rmax=" << rmax << ", sum_total=" << sum_total
                     << ", naxt_goal=" << next_goal << ", size=" << sketch->temp.data.size();
        }
      }
      rmin = rmax;
      wmin = w;
      last_fvalue = fvalue;
    } else {
      wmin += w;
    }
  }

  /*! \brief push final unfinished value to the sketch */
  inline void Finalize(unsigned max_size) {
    double rmax = rmin + wmin;
    auto &temp = sketch->temp.data;
    auto temp_data = temp.data();
    auto temp_size = temp.size();
    if (temp.empty() || last_fvalue > temp.back().value) {
      CHECK_LE(temp_size, max_size) << "Finalize: invalid maximum size, max_size=" << max_size
                                    << ", stemp.size=" << temp_size;
      // push to sketch
      temp_data[temp_size] =
          common::WQuantileSketch::Entry(static_cast<bst_float>(rmin), static_cast<bst_float>(rmax),
                                         static_cast<bst_float>(wmin), last_fvalue);
      sketch->temp.data = Span<Entry>{temp_data, temp_size + 1};
    }
    sketch->PushTemp();
  }
};

class SortedSketchContainer : public SketchContainerImpl {
 public:
  explicit SortedSketchContainer(Context const *ctx, int32_t max_bins,
                                 common::Span<FeatureType const> ft,
                                 std::vector<bst_idx_t> columns_size, bool use_group)
      : SketchContainerImpl{ctx, columns_size, max_bins, ft, use_group} {
    monitor_.Init(__func__);
    for (size_t i = 0; i < sketches_.size(); ++i) {
      auto eps = 2.0 / max_bins;
      sketches_[i].Init(columns_size_[i], eps);
    }
  }
  /**
   * \brief Push a sorted CSC page.
   */
  void PushColPage(SparsePage const &page, MetaInfo const &info, Span<float const> hessian);
};
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_QUANTILE_H_
