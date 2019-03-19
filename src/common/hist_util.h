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
#include "../tree/param.h"
#include "./quantile.h"
#include "./timer.h"
#include "../include/rabit/rabit.h"

namespace xgboost {

namespace tree
{
class RegTreeThreadSafe;
class SplitEvaluator;
}

namespace common {

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

 protected:
  virtual size_t SearchGroupIndFromBaseRow(
      std::vector<bst_uint> const& group_ptr, size_t const base_rowid) const;

  Monitor monitor_;
};

/*! \brief Builds the cut matrix on the GPU */
void DeviceSketch
  (const SparsePage& batch, const MetaInfo& info,
   const tree::TrainParam& param, HistCutMatrix* hmat, int gpu_batch_nrows);

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
  inline GHistRow operator[](bst_uint nid) const {
    std::lock_guard<std::mutex> lock(mutex_);
   return { const_cast<tree::GradStats*>(dmlc::BeginPtr(*data_arr_[nid])), nbins_};
  }

  // have we computed a histogram for i-th node?
  inline bool RowExists(bst_uint nid) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_init_[nid];
  }

  // initialize histogram collection
  inline void Init(uint32_t nbins) {
    std::lock_guard<std::mutex> lock(mutex_);
    for(size_t i=0; i < is_init_.size();++i)
      is_init_[i] = false;

    if (nbins_ != nbins)
    {
      for (size_t i = 0; i < data_arr_.size(); ++i) delete data_arr_[i];
      data_arr_.clear();
      nbins_ = nbins;
    }
  }

  ~HistCollection()
  {
    for (size_t i = 0; i < data_arr_.size(); ++i) delete data_arr_[i];
  }

  // create an empty histogram for i-th node
  inline void AddHistRow(bst_uint nid) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (data_arr_.size() <= nid)
    {
      data_arr_.resize(nid + 1, nullptr);
      is_init_.resize(nid + 1, false);
    }
    is_init_[nid] = true;

    if (data_arr_[nid] == nullptr) data_arr_[nid] = new std::vector<tree::GradStats>;

    if (data_arr_[nid]->size() == 0)
    {
      data_arr_[nid]->resize(nbins_);
    }
  }

  inline void AddHistRow(bst_uint nid1, bst_uint nid2) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (data_arr_.size() <= std::max(nid1, nid2))
    {
      data_arr_.resize(std::max(nid1, nid2) + 1, nullptr);
      is_init_.resize(std::max(nid1, nid2) + 1, false);
    }
    is_init_[nid1] = true;
    is_init_[nid2] = true;

    if (data_arr_[nid1] == nullptr) data_arr_[nid1] = new std::vector<tree::GradStats>;
    if (data_arr_[nid2] == nullptr) data_arr_[nid2] = new std::vector<tree::GradStats>;

    if (data_arr_[nid1]->size() == 0)
    {
      data_arr_[nid1]->resize(nbins_);
    }
    if (data_arr_[nid2]->size() == 0)
    {
      data_arr_[nid2]->resize(nbins_);
    }
  }


 private:
  /*! \brief number of all bins over all features */
  uint32_t nbins_ = 0;
  mutable std::mutex mutex_;
  std::vector<std::vector<tree::GradStats>*> data_arr_;
  std::vector<bool> is_init_;
};

/*!
 * \brief builder for histograms of gradient statistics
 */
template<typename TlsType>
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
                 GHistRow hist,
                 TlsType& hist_tls,
                 common::ColumnSampler& column_sampler,
                 tree::RegTreeThreadSafe& tree,
                 int32_t parent_nid,
                 const tree::TrainParam& param,
                 GHistRow sibling,
                 GHistRow parent,
                 int32_t this_nid,
                 int32_t another_nid,
                 const bool is_dense_layout,
                 const bool sync_hist);
  // same, with feature grouping
  void BuildBlockHist(const std::vector<GradientPair>& gpair,
                      const RowSetCollection::Elem row_indices,
                      const GHistIndexBlockMatrix& gmatb,
                      GHistRow hist);

  void SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent);

  uint32_t GetNumBins() {
      return nbins_;
  }

 private:
  /*! \brief number of threads for parallel computation */
  size_t nthread_;
  /*! \brief number of all bins over all features */
  uint32_t nbins_;
};


}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_UTIL_H_
