/*!
 * Copyright 2014 by Contributors
 * \file quantile.h
 * \brief util to compute quantiles
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_QUANTILE_H_
#define XGBOOST_COMMON_QUANTILE_H_

#include <dmlc/base.h>
#include <xgboost/logging.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace xgboost {
namespace common {
template <typename DType, typename RType>
class WXQSummary;

namespace detail {
/*! \brief an entry in the sketch summary */
template <typename RType, typename DType>
struct SketchEntry {
  /*! \brief minimum rank */
  RType rmin;
  /*! \brief maximum rank */
  RType rmax;
  /*! \brief maximum weight */
  RType wmin;
  /*! \brief the value of data */
  DType value;
  // constructor
  SketchEntry() = default;
  // constructor
  XGBOOST_DEVICE SketchEntry(RType rmin, RType rmax, RType wmin, DType value)
      : rmin(rmin), rmax(rmax), wmin(wmin), value(value) {}
  /*!
   * \brief debug function,  check Valid
   * \param eps the tolerate level for violating the relation
   */
  void CheckValid(RType eps = 0) const {
    CHECK(rmin >= 0 && rmax >= 0 && wmin >= 0) << "nonneg constraint";
    CHECK(rmax - rmin - wmin > -eps) << "relation constraint: min/max";
  }
  /*! \return rmin estimation for v strictly bigger than value */
  XGBOOST_DEVICE RType RMinNext() const { return rmin + wmin; }
  /*! \return rmax estimation for v strictly smaller than value */
  XGBOOST_DEVICE RType RMaxPrev() const { return rmax - wmin; }
};
/*! \brief input data queue before entering the summary */
template <typename RType, typename DType>
struct Queue {
  // entry in the queue
  struct QEntry {
    // value of the instance
    DType value;
    // weight of instance
    RType weight;
    // default constructor
    QEntry() = default;
    // constructor
    QEntry(DType value, RType weight) : value(value), weight(weight) {}
    // comparator on value
    bool operator<(const QEntry &b) const { return value < b.value; }
  };
  // the input queue
  std::vector<QEntry> queue;
  // end of the queue
  size_t qtail;
  // push data to the queue
  void Push(DType x, RType w) {
    if (qtail == 0 || queue[qtail - 1].value != x) {
      queue[qtail++] = QEntry(x, w);
    } else {
      queue[qtail - 1].weight += w;
    }
  }
  void MakeSummary(WXQSummary<RType, DType> *out);
};
}  // namespace detail
/*!
 * \brief experimental wsummary
 * \tparam DType type of data content
 * \tparam RType type of rank
 */
template <typename DType, typename RType>
class WXQSummary {
 public:
  using Entry = detail::SketchEntry<DType, RType>;
  /*! \brief data field */
  Entry *data;
  /*! \brief number of elements in the summary */
  size_t size;
  // constructor
  WXQSummary(Entry *data, size_t size) : data(data), size(size) {}
  /*!
   * \return the maximum error of the Summary
   */
  RType MaxError() const {
    RType res = data[0].rmax - data[0].rmin - data[0].wmin;
    for (size_t i = 1; i < size; ++i) {
      res = std::max(data[i].RMaxPrev() - data[i - 1].RMinNext(), res);
      res = std::max(data[i].rmax - data[i].rmin - data[i].wmin, res);
    }
    return res;
  }
  /*!
   * \brief debug function, validate whether the summary
   *  run consistency check to check if it is a valid summary
   * \param eps the tolerate error level, used when RType is floating point and
   *        some inconsistency could occur due to rounding error
   */
  void CheckValid(RType eps) const {
    for (size_t i = 0; i < size; ++i) {
      data[i].CheckValid(eps);
      if (i != 0) {
        CHECK(data[i].rmin >= data[i - 1].rmin + data[i - 1].wmin)
            << "rmin range constraint";
        CHECK(data[i].rmax >= data[i - 1].rmax + data[i].wmin)
            << "rmax range constraint";
      }
    }
  }
  // set prune
  void SetPrune(const WXQSummary<DType, RType> &src, size_t maxsize) {
    if (src.size <= maxsize) {
      this->CopyFrom(src);
      return;
    }
    RType begin = src.data[0].rmax;
    // n is number of points exclude the min/max points
    size_t n = maxsize - 2, nbig = 0;
    // these is the range of data exclude the min/max point
    RType range = src.data[src.size - 1].rmin - begin;
    // prune off zero weights
    if (range == 0.0f || maxsize <= 2) {
      // special case, contain only two effective data pts
      this->data[0] = src.data[0];
      this->data[1] = src.data[src.size - 1];
      this->size = 2;
      return;
    } else {
      range = std::max(range, static_cast<RType>(1e-3f));
    }
    // Get a big enough chunk size, bigger than range / n
    // (multiply by 2 is a safe factor)
    const RType chunk = 2 * range / n;
    // minimized range
    RType mrange = 0;
    {
      // first scan, grab all the big chunk
      // moving block index, exclude the two ends.
      size_t bid = 0;
      for (size_t i = 1; i < src.size - 1; ++i) {
        // detect big chunk data point in the middle
        // always save these data points.
        if (CheckLarge(src.data[i], chunk)) {
          if (bid != i - 1) {
            // accumulate the range of the rest points
            mrange += src.data[i].RMaxPrev() - src.data[bid].RMinNext();
          }
          bid = i;
          ++nbig;
        }
      }
      if (bid != src.size - 2) {
        mrange += src.data[src.size - 1].RMaxPrev() - src.data[bid].RMinNext();
      }
    }
    // assert: there cannot be more than n big data points
    if (nbig >= n) {
      // see what was the case
      LOG(INFO) << " check quantile stats, nbig=" << nbig << ", n=" << n;
      LOG(INFO) << " srcsize=" << src.size << ", maxsize=" << maxsize
                << ", range=" << range << ", chunk=" << chunk;
      src.Print();
      CHECK(nbig < n) << "quantile: too many large chunk";
    }
    this->data[0] = src.data[0];
    this->size = 1;
    // The counter on the rest of points, to be selected equally from small
    // chunks.
    n = n - nbig;
    // find the rest of point
    size_t bid = 0, k = 1, lastidx = 0;
    for (size_t end = 1; end < src.size; ++end) {
      if (end == src.size - 1 || CheckLarge(src.data[end], chunk)) {
        if (bid != end - 1) {
          size_t i = bid;
          RType maxdx2 = src.data[end].RMaxPrev() * 2;
          for (; k < n; ++k) {
            RType dx2 = 2 * ((k * mrange) / n + begin);
            if (dx2 >= maxdx2) break;
            while (i < end &&
                   dx2 >= src.data[i + 1].rmax + src.data[i + 1].rmin)
              ++i;
            if (i == end) break;
            if (dx2 < src.data[i].RMinNext() + src.data[i + 1].RMaxPrev()) {
              if (i != lastidx) {
                this->data[this->size++] = src.data[i];
                lastidx = i;
              }
            } else {
              if (i + 1 != lastidx) {
                this->data[this->size++] = src.data[i + 1];
                lastidx = i + 1;
              }
            }
          }
        }
        if (lastidx != end) {
          this->data[this->size++] = src.data[end];
          lastidx = end;
        }
        bid = end;
        // shift base by the gap
        begin += src.data[bid].RMinNext() - src.data[bid].RMaxPrev();
      }
    }
  }

  /*!
   * \brief set current summary to be merged summary of sa and sb
   * \param sa first input summary to be merged
   * \param sb second input summary to be merged
   */
  void SetCombine(const WXQSummary &sa, const WXQSummary &sb) {
    if (sa.size == 0) {
      this->CopyFrom(sb);
      return;
    }
    if (sb.size == 0) {
      this->CopyFrom(sa);
      return;
    }
    CHECK(sa.size > 0 && sb.size > 0);
    const Entry *a = sa.data, *a_end = sa.data + sa.size;
    const Entry *b = sb.data, *b_end = sb.data + sb.size;
    // extended rmin value
    RType aprev_rmin = 0, bprev_rmin = 0;
    Entry *dst = this->data;
    while (a != a_end && b != b_end) {
      // duplicated value entry
      if (a->value == b->value) {
        *dst = Entry(a->rmin + b->rmin, a->rmax + b->rmax, a->wmin + b->wmin,
                     a->value);
        aprev_rmin = a->RMinNext();
        bprev_rmin = b->RMinNext();
        ++dst;
        ++a;
        ++b;
      } else if (a->value < b->value) {
        *dst = Entry(a->rmin + bprev_rmin, a->rmax + b->RMaxPrev(), a->wmin,
                     a->value);
        aprev_rmin = a->RMinNext();
        ++dst;
        ++a;
      } else {
        *dst = Entry(b->rmin + aprev_rmin, b->rmax + a->RMaxPrev(), b->wmin,
                     b->value);
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
    this->size = dst - data;
    const RType tol = 10;
    RType err_mingap, err_maxgap, err_wgap;
    this->FixError(&err_mingap, &err_maxgap, &err_wgap);
    if (err_mingap > tol || err_maxgap > tol || err_wgap > tol) {
      LOG(INFO) << "mingap=" << err_mingap << ", maxgap=" << err_maxgap
                << ", wgap=" << err_wgap;
    }
    CHECK(size <= sa.size + sb.size) << "bug in combine";
  }
  // helper function to print the current content of sketch
  void Print() const {
    for (size_t i = 0; i < this->size; ++i) {
      LOG(CONSOLE) << "[" << i << "] rmin=" << data[i].rmin
                   << ", rmax=" << data[i].rmax << ", wmin=" << data[i].wmin
                   << ", v=" << data[i].value;
    }
  }
  /*!
   * \brief copy content from src
   * \param src source sketch
   */
  void CopyFrom(const WXQSummary &src) {
    size = src.size;
    std::memcpy(data, src.data, sizeof(Entry) * size);
  }

 private:
  // check if the block is large chunk
  static bool CheckLarge(const Entry &e, RType chunk) {
    return e.RMinNext() > e.RMaxPrev() + chunk;
  }
  // try to fix rounding error
  // and re-establish invariance
  void FixError(RType *err_mingap, RType *err_maxgap, RType *err_wgap) const {
    *err_mingap = 0;
    *err_maxgap = 0;
    *err_wgap = 0;
    RType prev_rmin = 0, prev_rmax = 0;
    for (size_t i = 0; i < this->size; ++i) {
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

/*!
 * \brief template for all quantile sketch algorithm
 *        that uses merge/prune scheme
 * \tparam DType type of data content
 * \tparam RType type of rank
 */
template <typename DType, typename RType>
class WXQuantileSketch {
 public:
  /*! \brief type of summary type */
  using Summary = WXQSummary<DType, RType>;
  /*! \brief the entry type */
  using Entry = typename Summary::Entry;
  /*! \brief same as summary, but use STL to backup the space */
  struct SummaryContainer : public Summary {
    std::vector<Entry> space;
    SummaryContainer(const SummaryContainer &src) : Summary(nullptr, src.size) {
      this->space = src.space;
      this->data = dmlc::BeginPtr(this->space);
    }
    SummaryContainer() : Summary(nullptr, 0) {}
    /*! \brief reserve space for summary */
    void Reserve(size_t size) {
      if (size > space.size()) {
        space.resize(size);
        this->data = dmlc::BeginPtr(space);
      }
    }
    /*!
     * \brief set the space to be merge of all Summary arrays
     * \param begin beginning position in the summary array
     * \param end ending position in the Summary array
     */
    void SetMerge(const Summary *begin, const Summary *end) {
      CHECK(begin < end) << "can not set combine to empty instance";
      size_t len = end - begin;
      if (len == 1) {
        this->Reserve(begin[0].size);
        this->CopyFrom(begin[0]);
      } else if (len == 2) {
        this->Reserve(begin[0].size + begin[1].size);
        this->SetMerge(begin[0], begin[1]);
      } else {
        // recursive merge
        SummaryContainer lhs, rhs;
        lhs.SetCombine(begin, begin + len / 2);
        rhs.SetCombine(begin + len / 2, end);
        this->Reserve(lhs.size + rhs.size);
        this->SetCombine(lhs, rhs);
      }
    }
    /*!
     * \brief do elementwise combination of summary array
     *        this[i] = combine(this[i], src[i]) for each i
     * \param src the source summary
     * \param max_nbyte maximum number of byte allowed in here
     */
    void Reduce(const Summary &src, size_t max_nbyte) {
      this->Reserve((max_nbyte - sizeof(this->size)) / sizeof(Entry));
      SummaryContainer temp;
      temp.Reserve(this->size + src.size);
      temp.SetCombine(*this, src);
      this->SetPrune(temp, space.size());
    }
    /*! \brief return the number of bytes this data structure cost in
     * serialization */
    static size_t CalcMemCost(size_t nentry) {
      return sizeof(size_t) + sizeof(Entry) * nentry;
    }
    /*! \brief save the data structure into stream */
    template <typename TStream>
    void Save(TStream &fo) const {  // NOLINT(*)
      fo.Write(&(this->size), sizeof(this->size));
      if (this->size != 0) {
        fo.Write(this->data, this->size * sizeof(Entry));
      }
    }
    /*! \brief load data structure from input stream */
    template <typename TStream>
    void Load(TStream &fi) {  // NOLINT(*)
      CHECK_EQ(fi.Read(&this->size, sizeof(this->size)), sizeof(this->size));
      this->Reserve(this->size);
      if (this->size != 0) {
        CHECK_EQ(fi.Read(this->data, this->size * sizeof(Entry)),
                 this->size * sizeof(Entry));
      }
    }
  };
  /*!
   * \brief initialize the quantile sketch, given the performance specification
   * \param maxn maximum number of data points can be feed into sketch
   * \param eps accuracy level of summary
   */
  WXQuantileSketch(size_t maxn, double eps) {
    LimitSizeLevel(maxn, eps, &nlevel, &limit_size);
    // lazy reserve the space, if there is only one value, no need to allocate
    // space
    inqueue_.queue.resize(1);
    inqueue_.qtail = 0;
  }

  static void LimitSizeLevel(size_t maxn, double eps, size_t *out_nlevel,
                             size_t *out_limit_size) {
    size_t &nlevel = *out_nlevel;
    size_t &limit_size = *out_limit_size;
    nlevel = 1;
    while (true) {
      limit_size = static_cast<size_t>(ceil(nlevel / eps)) + 1;
      size_t n = (1ULL << nlevel);
      if (n * limit_size >= maxn) break;
      ++nlevel;
    }
    // check invariant
    size_t n = (1ULL << nlevel);
    CHECK(n * limit_size >= maxn) << "invalid init parameter";
    CHECK(nlevel <= limit_size * eps) << "invalid init parameter";
  }

  /*!
   * \brief add an element to a sketch
   * \param x The element added to the sketch
   * \param w The weight of the element.
   */
  void Push(DType x, RType w = 1) {
    if (w == static_cast<RType>(0)) return;
    if (inqueue_.qtail == inqueue_.queue.size()) {
      // jump from lazy one value to limit_size * 2
      if (inqueue_.queue.size() == 1) {
        inqueue_.queue.resize(limit_size * 2);
      } else {
        temp.Reserve(limit_size * 2);
        inqueue_.MakeSummary(&temp);
        // cleanup queue
        inqueue_.qtail = 0;
        this->PushTemp();
      }
    }
    inqueue_.Push(x, w);
  }

  void PushSummary(const Summary &summary) {
    temp.Reserve(limit_size * 2);
    temp.SetPrune(summary, limit_size * 2);
    PushTemp();
  }

  /*! \brief push up temp */
  void PushTemp() {
    temp.Reserve(limit_size * 2);
    for (size_t l = 1; true; ++l) {
      this->InitLevel(l + 1);
      // check if level l is empty
      if (level[l].size == 0) {
        level[l].SetPrune(temp, limit_size);
        break;
      } else {
        // level 0 is actually temp space
        level[0].SetPrune(temp, limit_size);
        temp.SetCombine(level[0], level[l]);
        if (temp.size > limit_size) {
          // try next level
          level[l].size = 0;
        } else {
          // if merged record is still smaller, no need to send to next level
          level[l].CopyFrom(temp);
          break;
        }
      }
    }
  }
  /*! \brief get the summary after finalize */
  void GetSummary(SummaryContainer *out) {
    if (level.size() != 0) {
      out->Reserve(limit_size * 2);
    } else {
      out->Reserve(inqueue_.queue.size());
    }
    inqueue_.MakeSummary(out);
    if (level.size() != 0) {
      level[0].SetPrune(*out, limit_size);
      for (size_t l = 1; l < level.size(); ++l) {
        if (level[l].size == 0) continue;
        if (level[0].size == 0) {
          level[0].CopyFrom(level[l]);
        } else {
          out->SetCombine(level[0], level[l]);
          level[0].SetPrune(*out, limit_size);
        }
      }
      out->CopyFrom(level[0]);
    } else {
      if (out->size > limit_size) {
        temp.Reserve(limit_size);
        temp.SetPrune(*out, limit_size);
        out->CopyFrom(temp);
      }
    }
  }
  // used for debug, check if the sketch is valid
  void CheckValid(RType eps) const {
    for (size_t l = 1; l < level.size(); ++l) {
      level[l].CheckValid(eps);
    }
  }
  // initialize level space to at least nlevel
  void InitLevel(size_t nlevel) {
    if (level.size() >= nlevel) return;
    data.resize(limit_size * nlevel);
    level.resize(nlevel, Summary(nullptr, 0));
    for (size_t l = 0; l < level.size(); ++l) {
      level[l].data = dmlc::BeginPtr(data) + l * limit_size;
    }
  }

  // temporal summary, used for temp-merge
  SummaryContainer temp;

 private:
  // number of levels
  size_t nlevel;
  // size of summary in each level
  size_t limit_size;
  // the level of each summaries
  std::vector<Summary> level;
  // content of the summary
  std::vector<Entry> data;
  // input data queue
  detail::Queue<RType, DType> inqueue_;
};

namespace detail {
template <typename RType, typename DType>
void Queue<RType, DType>::MakeSummary(WXQSummary<RType, DType> *out) {
  using Entry = typename WXQSummary<RType, DType>::Entry;
  std::sort(queue.begin(), queue.begin() + qtail);
  out->size = 0;
  // start update sketch
  RType wsum = 0;
  // construct data with unique weights
  for (size_t i = 0; i < qtail;) {
    size_t j = i + 1;
    RType w = queue[i].weight;
    while (j < qtail && queue[j].value == queue[i].value) {
      w += queue[j].weight;
      ++j;
    }
    out->data[out->size++] = Entry(wsum, wsum + w, w, queue[i].value);
    wsum += w;
    i = j;
  }
}

};  // namespace detail
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_QUANTILE_H_
