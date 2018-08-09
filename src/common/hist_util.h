/*!
 * Copyright 2017 by Contributors
 * \file hist_util.h
 * \brief Utility for fast histogram aggregation
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_COMMON_HIST_UTIL_H_
#define XGBOOST_COMMON_HIST_UTIL_H_

#include <xgboost/data.h>
#include <limits>
#include <vector>
#include "row_set.h"
#include "../tree/fast_hist_param.h"
#include "../tree/param.h"
#include "./quantile.h"

namespace xgboost {

namespace common {

using tree::FastHistParam;

/*! \brief sums of gradient statistics corresponding to a histogram bin */
struct GHistEntry {
  /*! \brief sum of first-order gradient statistics */
  double sum_grad{0};
  /*! \brief sum of second-order gradient statistics */
  double sum_hess{0};

  GHistEntry()  = default;

  inline void Clear() {
    sum_grad = sum_hess = 0;
  }

  /*! \brief add a GradientPair to the sum */
  inline void Add(const GradientPair& e) {
    sum_grad += e.GetGrad();
    sum_hess += e.GetHess();
  }

  /*! \brief add a GHistEntry to the sum */
  inline void Add(const GHistEntry& e) {
    sum_grad += e.sum_grad;
    sum_hess += e.sum_hess;
  }

  /*! \brief set sum to be difference of two GHistEntry's */
  inline void SetSubtract(const GHistEntry& a, const GHistEntry& b) {
    sum_grad = a.sum_grad - b.sum_grad;
    sum_hess = a.sum_hess - b.sum_hess;
  }
};


/*! \brief Cut configuration for one feature */
struct HistCutUnit {
  /*! \brief the index pointer of each histunit */
  const bst_float* cut;
  /*! \brief number of cutting point, containing the maximum point */
  uint32_t size;
  // default constructor
  HistCutUnit() = default;
  // constructor
  HistCutUnit(const bst_float* cut, uint32_t size)
      : cut(cut), size(size) {}
};

/*! \brief cut configuration for all the features */
struct HistCutMatrix {
  /*! \brief unit pointer to rows by element position */
  std::vector<uint32_t> row_ptr;
  /*! \brief minimum value of each feature */
  std::vector<bst_float> min_val;
  /*! \brief the cut field */
  std::vector<bst_float> cut;
  uint32_t GetBinIdx(const Entry &e);
  /*! \brief Get histogram bound for fid */
  inline HistCutUnit operator[](bst_uint fid) const {
    return {dmlc::BeginPtr(cut) + row_ptr[fid],
                       row_ptr[fid + 1] - row_ptr[fid]};
  }

  using WXQSketch = common::WXQuantileSketch<bst_float, bst_float>;

  // create histogram cut matrix given statistics from data
  // using approximate quantile sketch approach
  void Init(DMatrix* p_fmat, uint32_t max_num_bins);

  void Init(std::vector<WXQSketch>* sketchs, uint32_t max_num_bins);
};

/*! \brief Builds the cut matrix on the GPU */
void DeviceSketch
  (const SparsePage& batch, const MetaInfo& info,
   const tree::TrainParam& param, HistCutMatrix* hmat);

/*!
 * \brief A single row in global histogram index.
 *  Directly represent the global index in the histogram entry.
 */
struct GHistIndexRow {
  /*! \brief The index of the histogram */
  const uint32_t* index;
  /*! \brief The size of the histogram */
  size_t size;
  GHistIndexRow() = default;
  GHistIndexRow(const uint32_t* index, size_t size)
      : index(index), size(size) {}
};

/*!
 * \brief preprocessed global index matrix, in CSR format
 *  Transform floating values to integer index in histogram
 *  This is a global histogram index.
 */
struct GHistIndexMatrix {
  /*! \brief row pointer to rows by element position */
  std::vector<size_t> row_ptr;
  /*! \brief The index data */
  std::vector<uint32_t> index;
  /*! \brief hit count of each index */
  std::vector<size_t> hit_count;
  /*! \brief The corresponding cuts */
  HistCutMatrix cut;
  // Create a global histogram matrix, given cut
  void Init(DMatrix* p_fmat, int max_num_bins);
  // get i-th row
  inline GHistIndexRow operator[](size_t i) const {
    return {&index[0] + row_ptr[i], row_ptr[i + 1] - row_ptr[i]};
  }
  inline void GetFeatureCounts(size_t* counts) const {
    auto nfeature = cut.row_ptr.size() - 1;
    for (unsigned fid = 0; fid < nfeature; ++fid) {
      auto ibegin = cut.row_ptr[fid];
      auto iend = cut.row_ptr[fid + 1];
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
    return {&index[0] + row_ptr[i], row_ptr[i + 1] - row_ptr[i]};
  }
};

class ColumnMatrix;

class GHistIndexBlockMatrix {
 public:
  void Init(const GHistIndexMatrix& gmat,
            const ColumnMatrix& colmat,
            const FastHistParam& param);

  inline GHistIndexBlock operator[](size_t i) const {
    return {blocks_[i].row_ptr_begin, blocks_[i].index_begin};
  }

  inline size_t GetNumBlock() const {
    return blocks_.size();
  }

 private:
  std::vector<size_t> row_ptr_;
  std::vector<uint32_t> index_;
  const HistCutMatrix* cut_;
  struct Block {
    const size_t* row_ptr_begin;
    const size_t* row_ptr_end;
    const uint32_t* index_begin;
    const uint32_t* index_end;
  };
  std::vector<Block> blocks_;
};

/*!
 * \brief histogram of graident statistics for a single node.
 *  Consists of multiple GHistEntry's, each entry showing total graident statistics 
 *     for that particular bin
 *  Uses global bin id so as to represent all features simultaneously
 */
struct GHistRow {
  /*! \brief base pointer to first entry */
  GHistEntry* begin;
  /*! \brief number of entries */
  uint32_t size;

  GHistRow() = default;
  GHistRow(GHistEntry* begin, uint32_t size)
    : begin(begin), size(size) {}
};

/*!
 * \brief histogram of gradient statistics for multiple nodes
 */
class HistCollection {
 public:
  // access histogram for i-th node
  inline GHistRow operator[](bst_uint nid) const {
    constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
    CHECK_NE(row_ptr_[nid], kMax);
    return {const_cast<GHistEntry*>(dmlc::BeginPtr(data_) + row_ptr_[nid]), nbins_};
  }

  // have we computed a histogram for i-th node?
  inline bool RowExists(bst_uint nid) const {
    const uint32_t k_max = std::numeric_limits<uint32_t>::max();
    return (nid < row_ptr_.size() && row_ptr_[nid] != k_max);
  }

  // initialize histogram collection
  inline void Init(uint32_t nbins) {
    nbins_ = nbins;
    row_ptr_.clear();
    data_.clear();
  }

  // create an empty histogram for i-th node
  inline void AddHistRow(bst_uint nid) {
    constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
    if (nid >= row_ptr_.size()) {
      row_ptr_.resize(nid + 1, kMax);
    }
    CHECK_EQ(row_ptr_[nid], kMax);

    row_ptr_[nid] = data_.size();
    data_.resize(data_.size() + nbins_);
  }

 private:
  /*! \brief number of all bins over all features */
  uint32_t nbins_;

  std::vector<GHistEntry> data_;

  /*! \brief row_ptr_[nid] locates bin for historgram of node nid */
  std::vector<size_t> row_ptr_;
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

  // construct a histogram via histogram aggregation
  void BuildHist(const std::vector<GradientPair>& gpair,
                 const RowSetCollection::Elem row_indices,
                 const GHistIndexMatrix& gmat,
                 const std::vector<bst_uint>& feat_set,
                 GHistRow hist);
  // same, with feature grouping
  void BuildBlockHist(const std::vector<GradientPair>& gpair,
                      const RowSetCollection::Elem row_indices,
                      const GHistIndexBlockMatrix& gmatb,
                      const std::vector<bst_uint>& feat_set,
                      GHistRow hist);
  // construct a histogram via subtraction trick
  void SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent);

 private:
  /*! \brief number of threads for parallel computation */
  size_t nthread_;
  /*! \brief number of all bins over all features */
  uint32_t nbins_;
  std::vector<GHistEntry> data_;
};


}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_UTIL_H_
