/*!
 * Copyright 2017 by Contributors
 * \file hist_util.h
 * \brief Utility for fast histogram aggregation
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_COMMON_HIST_UTIL_H_
#define XGBOOST_COMMON_HIST_UTIL_H_

#include <xgboost/data.h>
#include <xgboost/generic_parameters.h>
#include <limits>
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>

#include "row_set.h"
#include "../tree/param.h"
#include "./quantile.h"
#include "./timer.h"
#include "../include/rabit/rabit.h"
#include "random.h"

namespace xgboost {

/*!
 * \brief A C-style array with in-stack allocation. As long as the
 * array is smaller than MaxStackSize, it will be allocated inside the
 * stack. Otherwise, it will be heap-allocated.
 */
template<typename T, size_t MaxStackSize>
class MemStackAllocator {
 public:
  explicit MemStackAllocator(size_t required_size): required_size_(required_size) {
  }

  T* Get() {
    if (!ptr_) {
      if (MaxStackSize >= required_size_) {
        ptr_ = stack_mem_;
      } else {
        ptr_ =  reinterpret_cast<T*>(malloc(required_size_ * sizeof(T)));
        do_free_ = true;
      }
    }

    return ptr_;
  }

  ~MemStackAllocator() {
    if (do_free_) free(ptr_);
  }

 private:
  T* ptr_ = nullptr;
  bool do_free_ = false;
  size_t required_size_;
  T stack_mem_[MaxStackSize];
};

namespace common {

/*
 * \brief A thin wrapper around dynamically allocated C-style array.
 * Make sure to call resize() before use.
 */
template<typename T>
struct SimpleArray {
  ~SimpleArray() {
    free(ptr_);
    ptr_ = nullptr;
  }

  void resize(size_t n) {
    T* ptr = static_cast<T*>(malloc(n*sizeof(T)));
    memcpy(ptr, ptr_, n_ * sizeof(T));
    free(ptr_);
    ptr_ = ptr;
    n_ = n;
  }

  T& operator[](size_t idx) {
    return ptr_[idx];
  }

  T& operator[](size_t idx) const {
    return ptr_[idx];
  }

  size_t size() const {
    return n_;
  }

  T back() const {
    return ptr_[n_-1];
  }

  T* data() {
    return ptr_;
  }

  const T* data() const {
    return ptr_;
  }


  T* begin() {
    return ptr_;
  }

  const T* begin() const {
    return ptr_;
  }

  T* end() {
    return ptr_ + n_;
  }

  const T* end() const {
    return ptr_ + n_;
  }

 private:
  T* ptr_ = nullptr;
  size_t n_ = 0;
};

/*!
 * \brief A single row in global histogram index.
 *  Directly represent the global index in the histogram entry.
 */
using GHistIndexRow = Span<uint32_t const>;

// A CSC matrix representing histogram cuts, used in CPU quantile hist.
class HistogramCuts {
  friend class SparseCuts;
  friend class DenseCuts;
 protected:
  using BinIdx = uint32_t;
  common::Monitor monitor_;

  std::vector<bst_float> cut_values_;
  std::vector<uint32_t> cut_ptrs_;
  std::vector<float> min_vals_;

 public:
  HistogramCuts();
  HistogramCuts(HistogramCuts const& that) = delete;
  HistogramCuts(HistogramCuts&& that) noexcept(true) :
      monitor_{std::move(that.monitor_)},
      cut_values_{std::move(that.cut_values_)},
      cut_ptrs_{std::move(that.cut_ptrs_)},
      min_vals_{std::move(that.min_vals_)} {}
  HistogramCuts& operator=(HistogramCuts const& that) = delete;
  HistogramCuts& operator=(HistogramCuts&& that) noexcept(true) {
    monitor_ = std::move(that.monitor_);
    cut_ptrs_ = std::move(that.cut_ptrs_);
    cut_values_ = std::move(that.cut_values_);
    min_vals_ = std::move(that.min_vals_);
    return *this;
  }

  /* \brief Build histogram indices. */
  void Build(DMatrix* dmat, uint32_t const max_num_bins);
  /* \brief How many bins a feature has. */
  uint32_t FeatureBins(uint32_t feature) const {
    return cut_ptrs_.at(feature+1) - cut_ptrs_[feature];
  }

  std::vector<uint32_t> const& Ptrs()      const { return cut_ptrs_;   }
  std::vector<float>    const& Values()    const { return cut_values_; }
  std::vector<float>    const& MinValues() const { return min_vals_;   }

  size_t TotalBins() const { return cut_ptrs_.back(); }

  BinIdx SearchBin(float value, uint32_t column_id) {
    auto beg = cut_ptrs_.at(column_id);
    auto end = cut_ptrs_.at(column_id + 1);
    auto it = std::upper_bound(cut_values_.cbegin() + beg, cut_values_.cbegin() + end, value);
    if (it == cut_values_.cend()) {
      it = cut_values_.cend() - 1;
    }
    BinIdx idx = it - cut_values_.cbegin();
    return idx;
  }

  BinIdx SearchBin(Entry const& e) {
    return SearchBin(e.fvalue, e.index);
  }
};

class CutsBuilder {
 protected:
  HistogramCuts* p_cuts_;

 protected:
  /* \brief return whether group for ranking is used. */
  static bool UseGroup(DMatrix* dmat);

 public:
  explicit CutsBuilder(HistogramCuts* p_cuts) : p_cuts_{p_cuts} {}
  virtual ~CutsBuilder() = default;

  static uint32_t SearchGroupIndFromRow(
      std::vector<bst_uint> const& group_ptr, size_t const base_rowid) {
    using KIt = std::vector<bst_uint>::const_iterator;
    KIt res = std::lower_bound(group_ptr.cbegin(), group_ptr.cend() - 1, base_rowid);
    // Cannot use CHECK_NE because it will try to print the iterator.
    bool const found = res != group_ptr.cend() - 1;
    if (!found) {
      LOG(FATAL) << "Row " << base_rowid << " does not lie in any group!";
    }
    uint32_t group_ind = std::distance(group_ptr.cbegin(), res);
    return group_ind;
  }

  /* \brief Build histogram indices. */
  virtual void Build(DMatrix* dmat, uint32_t const max_num_bins) = 0;
};

class SparseCuts : public CutsBuilder {
  /* \brief Distrbute columns to each thread according to number of entries. */
  static std::vector<size_t> LoadBalance(SparsePage const& page, size_t const nthreads);
  Monitor monitor_;

 public:
  explicit SparseCuts(HistogramCuts* container) :
      CutsBuilder{container} {
    monitor_.Init(__func__);
  }

  /* \brief Concatonate the built cuts in each thread. */
  void Concat(std::vector<std::unique_ptr<SparseCuts>> const& cuts, uint32_t n_cols);
  /* \brief Build histogram indices in single thread. */
  void SingleThreadBuild(SparsePage const& page, MetaInfo const& info,
                         uint32_t max_num_bins,
                         bool const use_group_ind,
                         uint32_t beg, uint32_t end, uint32_t thread_id);
  void Build(DMatrix* dmat, uint32_t const max_num_bins) override;
};

/*! \brief Cut configuration for all the features. */
class DenseCuts  : public CutsBuilder {
 public:
  using WXQSketch = common::WXQuantileSketch<bst_float, bst_float>;

  explicit DenseCuts(HistogramCuts* container) :
      CutsBuilder{container} {
    monitor_.Init(__func__);
  }
  void Init(std::vector<WXQSketch>* sketchs, uint32_t max_num_bins);
  void Build(DMatrix* p_fmat, uint32_t max_num_bins) override;

 protected:
  Monitor monitor_;
};

// FIXME(trivialfis): Merge this into generic cut builder.
/*! \brief Builds the cut matrix on the GPU.
 *
 *  \return The row stride across the entire dataset.
 */
size_t DeviceSketch
  (const tree::TrainParam& param, const LearnerTrainParam &learner_param, int gpu_batch_nrows,
   DMatrix* dmat, HistogramCuts* hmat);


/*!
 * \brief preprocessed global index matrix, in CSC format
 *  Transform floating values to integer index in histogram
 *  This is a global histogram index.
 */
struct GHistIndexMatrix {
  /*! \brief row pointer to rows by element position */
  // std::vector<size_t> row_ptr;
  SimpleArray<size_t> row_ptr;
  /*! \brief The index data */
  SimpleArray<uint32_t> index;
  /*! \brief hit count of each index */
  std::vector<size_t> hit_count;
  /*! \brief The corresponding cuts */
  HistogramCuts cut;
  // Create a global histogram matrix, given cut
  void Init(DMatrix* p_fmat, int max_num_bins);
  // get i-th row
  inline GHistIndexRow operator[](size_t i) const {
    return {&index[0] + row_ptr[i],
            static_cast<GHistIndexRow::index_type>(
                row_ptr[i + 1] - row_ptr[i])};
  }
  inline void GetFeatureCounts(size_t* counts) const {
    auto nfeature = cut.Ptrs().size() - 1;
    for (unsigned fid = 0; fid < nfeature; ++fid) {
      auto ibegin = cut.Ptrs()[fid];
      auto iend = cut.Ptrs()[fid + 1];
      for (auto i = ibegin; i < iend; ++i) {
        counts[fid] += hit_count[i];
      }
    }
  }

 private:
  std::vector<size_t> hit_count_tloc_;
};

struct GHistIndexBlock {
  const size_t* row_ptr;
  const uint32_t* index;

  inline GHistIndexBlock(const size_t* row_ptr, const uint32_t* index)
    : row_ptr(row_ptr), index(index) {}

  // get i-th row
  inline GHistIndexRow operator[](size_t i) const {
    return {&index[0] + row_ptr[i], detail::ptrdiff_t(row_ptr[i + 1] - row_ptr[i])};
  }
};

class ColumnMatrix;

class GHistIndexBlockMatrix {
 public:
  void Init(const GHistIndexMatrix& gmat,
            const ColumnMatrix& colmat,
            const tree::TrainParam& param);

  inline GHistIndexBlock operator[](size_t i) const {
    return {blocks_[i].row_ptr_begin, blocks_[i].index_begin};
  }

  inline size_t GetNumBlock() const {
    return blocks_.size();
  }

 private:
  std::vector<size_t> row_ptr_;
  std::vector<uint32_t> index_;
  const HistogramCuts* cut_;
  struct Block {
    const size_t* row_ptr_begin;
    const size_t* row_ptr_end;
    const uint32_t* index_begin;
    const uint32_t* index_end;
  };
  std::vector<Block> blocks_;
};

/*!
 * \brief used instead of GradStats to have float instead of double to reduce histograms
 * this improves performance by 10-30% and memory consumption for histograms by 2x
 * accuracy in both cases is the same
 */
struct GradStatHist {
  typedef float GradType;
  /*! \brief sum gradient statistics */
  GradType sum_grad;
  /*! \brief sum hessian statistics */
  GradType sum_hess;

  GradStatHist() : sum_grad{0}, sum_hess{0} {
    static_assert(sizeof(GradStatHist) == 8,
                  "Size of GradStatHist is not 8 bytes.");
  }

  inline void Add(const GradStatHist& b) {
    sum_grad += b.sum_grad;
    sum_hess += b.sum_hess;
  }

  inline void Add(const tree::GradStats& b) {
    sum_grad += b.sum_grad;
    sum_hess += b.sum_hess;
  }

  inline void Add(const GradientPair& p) {
    this->Add(p.GetGrad(), p.GetHess());
  }

  inline void Add(const GradType& grad, const GradType& hess) {
    sum_grad += grad;
    sum_hess += hess;
  }

  inline tree::GradStats ToGradStat() const {
    return tree::GradStats(sum_grad, sum_hess);
  }

  inline void SetSubstract(const GradStatHist& a, const GradStatHist& b) {
    sum_grad = a.sum_grad - b.sum_grad;
    sum_hess = a.sum_hess - b.sum_hess;
  }

  inline void SetSubstract(const tree::GradStats& a, const GradStatHist& b) {
    sum_grad = a.sum_grad - b.sum_grad;
    sum_hess = a.sum_hess - b.sum_hess;
  }

  inline GradType GetGrad() const { return sum_grad; }
  inline GradType GetHess() const { return sum_hess; }
  inline static void Reduce(GradStatHist& a, const GradStatHist& b) { // NOLINT(*)
    a.Add(b);
  }
};

using GHistRow = Span<GradStatHist>;

/*!
 * \brief histogram of gradient statistics for multiple nodes
 */
class HistCollection {
 public:
  // access histogram for i-th node
  inline GHistRow operator[](bst_uint nid) {
    AddHistRow(nid);
    return { const_cast<GradStatHist*>(dmlc::BeginPtr(data_arr_[nid])), nbins_};
  }

  // have we computed a histogram for i-th node?
  inline bool RowExists(bst_uint nid) const {
    return nid < data_arr_.size();
  }

  // initialize histogram collection
  inline void Init(uint32_t nbins) {
    if (nbins_ != nbins) {
      data_arr_.clear();
      nbins_ = nbins;
    }
  }

  // create an empty histogram for i-th node
  inline void AddHistRow(bst_uint nid) {
    if (data_arr_.size() <= nid) {
      size_t prev = data_arr_.size();
      data_arr_.resize(nid + 1);

      for (size_t i = prev; i < data_arr_.size(); ++i) {
        data_arr_[i].resize(nbins_);
      }
    }
  }

 private:
  /*! \brief number of all bins over all features */
  uint32_t nbins_ = 0;
  std::vector<std::vector<GradStatHist>> data_arr_;
};


/*!
 * \brief builder for histograms of gradient statistics
 */
class GHistBuilder {
 public:
  // initialize builder
  inline void Init(size_t nthread, uint32_t nbins) {
    nthread_ = nthread;
    nbins_ = nbins;
  }

  void BuildBlockHist(const std::vector<GradientPair>& gpair,
                                    const RowSetCollection::Elem row_indices,
                                    const GHistIndexBlockMatrix& gmatb,
                                    GHistRow hist) {
    constexpr int kUnroll = 8;  // loop unrolling factor
    const int32_t nblock = gmatb.GetNumBlock();
    const size_t nrows = row_indices.end - row_indices.begin;
    const size_t rest = nrows % kUnroll;

    #pragma omp parallel for
    for (int32_t bid = 0; bid < nblock; ++bid) {
      auto gmat = gmatb[bid];

      for (size_t i = 0; i < nrows - rest; i += kUnroll) {
        size_t rid[kUnroll];
        size_t ibegin[kUnroll];
        size_t iend[kUnroll];
        GradientPair stat[kUnroll];
        for (int k = 0; k < kUnroll; ++k) {
          rid[k] = row_indices.begin[i + k];
        }
        for (int k = 0; k < kUnroll; ++k) {
          ibegin[k] = gmat.row_ptr[rid[k]];
          iend[k] = gmat.row_ptr[rid[k] + 1];
        }
        for (int k = 0; k < kUnroll; ++k) {
          stat[k] = gpair[rid[k]];
        }
        for (int k = 0; k < kUnroll; ++k) {
          for (size_t j = ibegin[k]; j < iend[k]; ++j) {
            const uint32_t bin = gmat.index[j];
            hist[bin].Add(stat[k]);
          }
        }
      }
      for (size_t i = nrows - rest; i < nrows; ++i) {
        const size_t rid = row_indices.begin[i];
        const size_t ibegin = gmat.row_ptr[rid];
        const size_t iend = gmat.row_ptr[rid + 1];
        const GradientPair stat = gpair[rid];
        for (size_t j = ibegin; j < iend; ++j) {
          const uint32_t bin = gmat.index[j];
          hist[bin].Add(stat);
        }
      }
    }
  }

  uint32_t GetNumBins() {
      return nbins_;
  }

 private:
  /*! \brief number of threads for parallel computation */
  size_t nthread_;
  /*! \brief number of all bins over all features */
  uint32_t nbins_;
};


void BuildHistLocalDense(size_t istart, size_t iend, size_t nrows, const size_t* rid,
    const uint32_t* index, const GradientPair::ValueT* pgh, const size_t* row_ptr,
    GradStatHist::GradType* data_local_hist, GradStatHist* grad_stat);

void BuildHistLocalSparse(size_t istart, size_t iend, size_t nrows, const size_t* rid,
    const uint32_t* index, const GradientPair::ValueT* pgh, const size_t* row_ptr,
    GradStatHist::GradType* data_local_hist, GradStatHist* grad_stat);

void SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent);

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_UTIL_H_
