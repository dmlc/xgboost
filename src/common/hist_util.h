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
#include <memory>
#include <atomic>
#include "row_set.h"
#include "../tree/param.h"
#include "./quantile.h"
#include "./timer.h"
#include "../include/rabit/rabit.h"

namespace xgboost {
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




/*! \brief Cut configuration for all the features. */
struct HistCutMatrix {
  /*! \brief Unit pointer to rows by element position */
  std::vector<uint32_t> row_ptr;
  /*! \brief minimum value of each feature */
  std::vector<bst_float> min_val;
  /*! \brief the cut field */
  std::vector<bst_float> cut;
  uint32_t GetBinIdx(const Entry &e);

  using WXQSketch = common::WXQuantileSketch<bst_float, bst_float>;

  // create histogram cut matrix given statistics from data
  // using approximate quantile sketch approach
  void Init(DMatrix* p_fmat, uint32_t max_num_bins);

  void Init(std::vector<WXQSketch>* sketchs, uint32_t max_num_bins);

  HistCutMatrix();
  size_t NumBins() const { return row_ptr.back(); }

 protected:
  virtual size_t SearchGroupIndFromBaseRow(
      std::vector<bst_uint> const& group_ptr, size_t const base_rowid) const;

  Monitor monitor_;
};

/*!
 * \brief A container that holds the device sketches across all
 *  sparse page batches which are distributed to different devices.
 *  As sketches are aggregated by column, the atomic flag simulates
 *  a lock in user land when multiple devices could be pushing
 *  sketch summary for the same column across distinct rows.
 */
struct SketchContainer {
  std::vector<HistCutMatrix::WXQSketch> sketches_;
  std::vector<std::unique_ptr<std::atomic_flag>> col_locks_;
};

/*! \brief Builds the cut matrix on the GPU */
void DeviceSketch
  (const SparsePage& batch, const MetaInfo& info,
   const tree::TrainParam& param, SketchContainer *sketches,
   int gpu_batch_nrows);


/*!
 * \brief A single row in global histogram index.
 *  Directly represent the global index in the histogram entry.
 */
using GHistIndexRow = Span<uint32_t const>;

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
    return {&index[0] + row_ptr[i],
            static_cast<GHistIndexRow::index_type>(
                row_ptr[i + 1] - row_ptr[i])};
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
 *  Consists of multiple GradStats, each entry showing total graident statistics
 *     for that particular bin
 *  Uses global bin id so as to represent all features simultaneously
 */
using GHistRow = Span<tree::GradStats>;

/*!
 * \brief histogram of gradient statistics for multiple nodes
 */
class HistCollection {
 public:
  // access histogram for i-th node
  GHistRow operator[](bst_uint nid) const {
    constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
    CHECK_NE(row_ptr_[nid], kMax);
    tree::GradStats* ptr =
        const_cast<tree::GradStats*>(dmlc::BeginPtr(data_) + row_ptr_[nid]);
    return {ptr, nbins_};
  }

  // have we computed a histogram for i-th node?
  bool RowExists(bst_uint nid) const {
    const uint32_t k_max = std::numeric_limits<uint32_t>::max();
    return (nid < row_ptr_.size() && row_ptr_[nid] != k_max);
  }

  // initialize histogram collection
  void Init(uint32_t nbins) {
    nbins_ = nbins;
    row_ptr_.clear();
    data_.clear();
  }

  // create an empty histogram for i-th node
  void AddHistRow(bst_uint nid) {
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

  std::vector<tree::GradStats> data_;

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
    thread_init_.resize(nthread_);
  }

  // construct a histogram via histogram aggregation
  void BuildHist(const std::vector<GradientPair>& gpair,
                 const RowSetCollection::Elem row_indices,
                 const GHistIndexMatrix& gmat,
                 GHistRow hist);
  // same, with feature grouping
  void BuildBlockHist(const std::vector<GradientPair>& gpair,
                      const RowSetCollection::Elem row_indices,
                      const GHistIndexBlockMatrix& gmatb,
                      GHistRow hist);
  // construct a histogram via subtraction trick
  void SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent);

  uint32_t GetNumBins() {
      return nbins_;
  }

 private:
  /*! \brief number of threads for parallel computation */
  size_t nthread_;
  /*! \brief number of all bins over all features */
  uint32_t nbins_;
  std::vector<size_t> thread_init_;
  std::vector<tree::GradStats> data_;
};


}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_UTIL_H_
