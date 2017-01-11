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

namespace xgboost {
namespace common {

/*! \brief sums of gradient statistics corresponding to a histogram bin */
struct GHistEntry {
  /*! \brief sum of first-order gradient statistics */
  double sum_grad;
  /*! \brief sum of second-order gradient statistics */
  double sum_hess;

  GHistEntry() : sum_grad(0), sum_hess(0) {}

  /*! \brief add a bst_gpair to the sum */
  inline void Add(const bst_gpair& e) {
    sum_grad += e.grad;
    sum_hess += e.hess;
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
  const bst_float *cut;
  /*! \brief number of cutting point, containing the maximum point */
  size_t size;
  // default constructor
  HistCutUnit() {}
  // constructor
  HistCutUnit(const bst_float *cut, unsigned size)
      : cut(cut), size(size) {}
};

/*! \brief cut configuration for all the features */
struct HistCutMatrix {
  /*! \brief actual unit pointer */
  std::vector<unsigned> row_ptr;
  /*! \brief minimum value of each feature */
  std::vector<bst_float> min_val;
  /*! \brief the cut field */
  std::vector<bst_float> cut;
  /*! \brief Get histogram bound for fid */
  inline HistCutUnit operator[](unsigned fid) const {
    return HistCutUnit(dmlc::BeginPtr(cut) + row_ptr[fid],
                       row_ptr[fid + 1] - row_ptr[fid]);
  }
  // create histogram cut matrix given statistics from data
  // using approximate quantile sketch approach
  void Init(DMatrix *p_fmat, size_t max_num_bins);
};


/*!
 * \brief A single row in global histogram index.
 *  Directly represent the global index in the histogram entry.
 */
struct GHistIndexRow {
  /*! \brief The index of the histogram */
  const unsigned *index;
  /*! \brief The size of the histogram */
  unsigned size;
  GHistIndexRow() {}
  GHistIndexRow(const unsigned* index, unsigned size)
      : index(index), size(size) {}
};

/*!
 * \brief preprocessed global index matrix, in CSR format
 *  Transform floating values to integer index in histogram
 *  This is a global histogram index.
 */
struct GHistIndexMatrix {
  /*! \brief row pointer */
  std::vector<unsigned> row_ptr;
  /*! \brief The index data */
  std::vector<unsigned> index;
  /*! \brief hit count of each index */
  std::vector<unsigned> hit_count;
  /*! \brief optional remap index from outter row_id -> internal row_id*/
  std::vector<unsigned> remap_index;
  /*! \brief The corresponding cuts */
  const HistCutMatrix* cut;
  // Create a global histogram matrix, given cut
  void Init(DMatrix *p_fmat);
  // build remap
  void Remap();
  // get i-th row
  inline GHistIndexRow operator[](bst_uint i) const {
    return GHistIndexRow(
        &index[0] + row_ptr[i], row_ptr[i + 1] - row_ptr[i]);
  }
};

/*!
 * \brief histogram of graident statistics for a single node.
 *  Consists of multiple GHistEntry's, each entry showing total graident statistics 
 *     for that particular bin
 *  Uses global bin id so as to represent all features simultaneously
 */
struct GHistRow {
  /*! \brief base pointer to first entry */
  GHistEntry *begin;
  /*! \brief number of entries */
  unsigned size;

  GHistRow() {}
  GHistRow(GHistEntry *begin, unsigned size)
    : begin(begin), size(size) {}
};

/*!
 * \brief histogram of gradient statistics for multiple nodes
 */
class HistCollection {
 public:
  // access histogram for i-th node
  inline GHistRow operator[](bst_uint nid) const {
    const size_t kMax = std::numeric_limits<size_t>::max();
    CHECK_NE(row_ptr_[nid], kMax);
    return GHistRow(const_cast<GHistEntry*>(dmlc::BeginPtr(data_) + row_ptr_[nid]), nbins_);
  }

  // have we computed a histogram for i-th node?
  inline bool RowExists(bst_uint nid) const {
    const size_t kMax = std::numeric_limits<size_t>::max();
    return (nid < row_ptr_.size() && row_ptr_[nid] != kMax);
  }

  // initialize histogram collection
  inline void Init(size_t nbins) {
    nbins_ = nbins;

    row_ptr_.clear();
    data_.clear();
  }

  // create an empty histogram for i-th node
  inline void AddHistRow(bst_uint nid) {
    const size_t kMax = std::numeric_limits<size_t>::max();

    if (nid >= row_ptr_.size()) {
      row_ptr_.resize(nid + 1, kMax);
    }

    CHECK_EQ(row_ptr_[nid], kMax);

    row_ptr_[nid] = data_.size();
    data_.resize(data_.size() + nbins_);
  }

 private:
  /*! \brief number of all bins over all features */
  size_t nbins_;

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
  inline void Init(size_t nthread, size_t nbins) {
    nthread_ = nthread;
    nbins_ = nbins;
    data_.resize(nthread * nbins, GHistEntry());
  }

  // construct a histogram via histogram aggregation
  void BuildHist(const std::vector<bst_gpair>& gpair,
                 const RowSetCollection::Elem row_indices,
                 const GHistIndexMatrix& gmat,
                 GHistRow hist);
  // construct a histogram via subtraction trick
  void SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent);

 private:
  /*! \brief number of threads for parallel computation */
  size_t nthread_;
  /*! \brief number of all bins over all features */
  size_t nbins_;
  std::vector<GHistEntry> data_;
};


}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_UTIL_H_
