#ifndef XGBOOST_UTILS_QUANTILE_H_
#define XGBOOST_UTILS_QUANTILE_H_
/*!
 * \file quantile
 * \brief util to compute quantiles 
 * \author Tianqi Chen
 */
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
#include "./utils.h"

namespace xgboost {
namespace utils {
/*! 
 * \brief a helper class to compute streaming quantile
 * \tparam DType type of data content
 * \tparam RType type of rank
 */
template<typename DType, typename RType=unsigned>
class QuantileSketch {
 public:
  /*! \brief an entry in the sketch summary */
  struct Entry {
    /*! \brief minimum rank */
    RType rmin;
    /*! \brief maximum rank */
    RType rmax;
    /*! \brief the value of data */
    DType value;
    // constructor
    Entry(void) {}
    // constructor
    Entry(RType rmin, RType rmax, DType value)
        : rmin(rmin), rmax(rmax), value(value) {}
  };
  /*! 
   * \brief this is data structure presenting one summary
   */
  struct Summary {
    /*! \brief data field */
    Entry *data;
    /*! \brief number of elements in the summary */
    RType size;
    /*! \brief the maximum error of the summary */
    inline RType MaxError(void) const {
      RType res = 0;
      for (RType i = 1; i < size; ++i) {
        res = std::max(data[i].rmax - data[i-1].rmin, res);
      }
      return res;
    }
    /*! \return maximum rank in the summary */
    inline RType MaxRank(void) const {
      return data[size - 1].rmax;
    }
    /*! \brief set size to 0 */
    inline void Clear(void) {
      size = 0;
    }
    /*! 
     * \brief copy content from src
     * \param src source sketch
     */
    inline void CopyFrom(const Summary &src) {
      size = src.size;
      std::memcpy(data, src.data, sizeof(Entry) * size);
    }
    /*! 
     * \brief set current summary to be pruned summary of src
     *        assume data field is already allocated to be at least maxsize
     * \param src source summary
     * \param maxsize size we can afford in the pruned sketch
     */
    inline void SetPrune(const Summary &src, RType maxsize) {
      const RType max_rank = src.MaxRank();
      this->size = maxsize;
      data[0] = src.data[0];
      RType n = maxsize - 1;
      RType top = 1;
      for (RType i = 1; i < n; ++i) {
        RType k = (i * max_rank) / n;
        while (k > src.data[top + 1].rmax) ++top;
        // assert src.data[top].rmin <= k
        // because k > src.data[top].rmax >= src.data[top].rmin
        if ((k - src.data[top].rmin) < (src.data[top+1].rmax - k)) {
          data[i] = src.data[top];
        } else {
          data[i] = src.data[top + 1];
        }
      }
      data[n] = src.data[src.size - 1];
    }
    inline void SetCombine(const Summary &sa,
                           const Summary &sb) {
      utils::Assert(sa.size > 0 && sb.size > 0, "invalid input for merge"); 
      const Entry *a = sa.data, *a_end = sa.data + sa.size;
      const Entry *b = sb.data, *b_end = sb.data + sb.size;
      this->size = sa.size + sb.size;
      RType aprev_rmin = 0, bprev_rmin = 0;
      Entry *dst = this->data;
      while (a != a_end && b != b_end) {
        if (a->value < b->value) {
          *dst = Entry(bprev_rmin + a->rmin,
                       a->rmax + b->rmax - 1, a->value);
          aprev_rmin = a->rmin;
          ++dst; ++a;
        } else {
          *dst = Entry(aprev_rmin + b->rmin, 
                       b->rmax + a->rmax - 1, b->value);
          bprev_rmin = b->rmin;
          ++dst; ++b;
        }
      }
      if (a != a_end) {
        RType bprev_rmax = (b_end - 1)->rmax;
        do {
          *dst = Entry(bprev_rmin + a->rmin, bprev_rmax + a->rmax, a->value);
          ++dst; ++a;
        } while (a != a_end);
      }
      if (b != b_end) {
        RType aprev_rmax = (a_end - 1)->rmax;
        do {
          *dst = Entry(aprev_rmin + b->rmin, aprev_rmax + b->rmax, b->value);
          ++dst; ++b;
        } while (b != b_end);
      }
      utils::Assert(dst == data + size, "bug in combine");
    }
  };
  // same as summary, but use STL to backup the space
  struct SummaryContainer : public Summary {
    std::vector<Entry> space;
    /*! \brief reserve space for summary */
    inline void Reserve(size_t size) {
      space.resize(size);
      this->data = BeginPtr(space);
    }
    /*! 
     * \brief set the space to be merge of all Summary arrays
     * \param begin begining position in th summary array
     * \param end ending position in the Summary array
     */
    inline void SetMerge(const Summary *begin,
                         const Summary *end) {
      utils::Assert(begin < end, "can not set combine to empty instance");
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
  };
  /*! 
   * \brief intialize the quantile sketch, given the performance specification
   * \param maxn maximum number of data points can be encountered
   * \param eps accuracy level of summary
   */
  inline void Init(RType maxn, double eps) {
    eps  = eps * 0.5;
    size_t L = 0;
    size_t b = std::max(floor(log2(eps * maxn) / eps), 8.0);
    // check for lower 
    while (b < maxn) {
      L = ceil(log2(maxn / b)) + 1;
      if (L < eps * b) break;
      ++b;
    }
    L += 1;
    inqueue.resize(b);
    level_batch = (b + 1) / 2 + 1;
    temp.Reserve(level_batch * 2);
    data.resize(level_batch * L);
    for (size_t l = 0; l < L; ++l) {
      Summary s; s.size = 0;
      s.data = BeginPtr(data) + l * level_batch;
      level.push_back(s);
    }
    printf("init L = %lu, b = %lu, %lu size\n",L, b, data.size());
    qtail = 0;
  }
  /*! 
   * \brief add an element to a sketch 
   * \param x the elemented added to the sketch
   */
  inline void Add(DType x) {
    inqueue[qtail++] = x;
    if (qtail == inqueue.size()) {
      // start update sketch
      std::sort(inqueue.begin(), inqueue.end());
      for (size_t i = 0; i < qtail; ++i) {
        temp.data[i] = Entry(i + 1, i + 1, inqueue[i]);
      }
      temp.size = static_cast<RType>(qtail);
      // clean up queue
      qtail = 0;
      for (size_t l = 1; l < level.size(); ++l) {
        // check if level l is empty
        if (level[l].size == 0) {
          level[l].SetPrune(temp, level_batch);
          return;
        } else {
          // level 0 is actually temp space
          level[0].SetPrune(temp, level_batch);
          temp.SetCombine(level[0], level[l]);
          level[l].size = 0;
        }
      }
      utils::Error("adding more element than allowed");
    }
  }
  /*! 
   * \brief finalize the result after all data has been passed 
   *        copy the final result to level 0
   *        this can only be called once
   */
  inline void Finalize(void) {
    // start update sketch
    std::sort(inqueue.begin(), inqueue.begin() + qtail);
    for (size_t i = 0; i < qtail; ++i) {
      temp.data[i] = Entry(i + 1, i + 1, inqueue[i]);
    }
    temp.size = static_cast<RType>(qtail);
    if (temp.size < level_batch) {
      level[0].CopyFrom(temp);
    } else {
      level[0].SetPrune(temp, level_batch);
    }
    // start adding other things in
    for (size_t l = 1; l < level.size(); ++l) {
      if (level[l].size == 0) continue;
      if (level[0].size == 0) {
        level[0].CopyFrom(level[l]);
      } else {
        temp.SetCombine(level[0], level[l]);
        level[0].SetPrune(temp, level_batch);        
      }
      level[l].size = 0;
    }
  }
  /*! \brief get the summary after finalize */
  inline Summary GetSummary(void) const {
    return level[0];
  }  
  
 private:  
  // the input queue
  std::vector<DType> inqueue;
  // end of the queue
  size_t qtail;
  // size of summary in each level
  size_t level_batch;
  // content of the summary
  std::vector<Entry> data;
  // different level of summary
  std::vector<Summary> level;  
  // temporal summary, used for temp-merge
  SummaryContainer temp;  
};
}  // utils
}  // xgboost
#endif
