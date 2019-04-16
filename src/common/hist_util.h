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
#include <algorithm>
#include <utility>
#include "row_set.h"
#include "../tree/param.h"
#include "./quantile.h"
#include "./timer.h"
#include "../include/rabit/rabit.h"
#include "random.h"

#if defined(XGBOOST_MM_PREFETCH_PRESENT)
  #include <xmmintrin.h>
  #define PREFETCH_READ_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
  #define PREFETCH_READ_T0(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 3)
#else  // no SW pre-fetching available; PREFETCH_READ_T0 is no-op
  #define PREFETCH_READ_T0(addr) do {} while (0)
#endif  // defined(XGBOOST_MM_PREFETCH_PRESENT)

namespace xgboost {

/*!
 * \brief A C-style array with in-stack allocation. As long as the array is smaller than MaxStackSize, it will be allocated inside the stack. Otherwise, it will be heap-allocated.
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

template<typename Func>
inline void ParallelFor(const size_t n, Func func) {
  if (n) {
    #pragma omp taskgroup
    {
      for (size_t iblock = 0; iblock < n; iblock++) {
        #pragma omp task
        func(iblock);
      }
    }
  }
}

template<typename Func>
inline void SeqFor(const size_t n, Func func) {
  for (size_t iblock = 0; iblock < n; iblock++) {
    func(iblock);
  }
}

namespace tree {
class SplitEvaluator;
}

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
  // std::vector<size_t> row_ptr;
  SimpleArray<size_t> row_ptr;
  /*! \brief The index data */
  SimpleArray<uint32_t> index;
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
    for (size_t i = 0; i < is_init_.size(); ++i)
      is_init_[i] = false;

    if (nbins_ != nbins) {
      for (size_t i = 0; i < data_arr_.size(); ++i) {
        delete data_arr_[i];
      }
      data_arr_.clear();
      nbins_ = nbins;
    }
  }

  ~HistCollection() {
    for (size_t i = 0; i < data_arr_.size(); ++i) {
      delete data_arr_[i];
    }
  }

  // create an empty histogram for i-th node
  inline void AddHistRow(bst_uint nid) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (data_arr_.size() <= nid) {
      data_arr_.resize(nid + 1, nullptr);
      is_init_.resize(nid + 1, false);
    }
    is_init_[nid] = true;

    if (data_arr_[nid] == nullptr) {
      data_arr_[nid] = new std::vector<tree::GradStats>;
    }

    if (data_arr_[nid]->size() == 0) {
      data_arr_[nid]->resize(nbins_);
    }
  }

  inline void AddHistRow(bst_uint nid1, bst_uint nid2) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (data_arr_.size() <= std::max(nid1, nid2)) {
      data_arr_.resize(std::max(nid1, nid2) + 1, nullptr);
      is_init_.resize(std::max(nid1, nid2) + 1, false);
    }
    is_init_[nid1] = true;
    is_init_[nid2] = true;

    if (data_arr_[nid1] == nullptr) {
      data_arr_[nid1] = new std::vector<tree::GradStats>;
    }
    if (data_arr_[nid2] == nullptr) {
      data_arr_[nid2] = new std::vector<tree::GradStats>;
    }
    if (data_arr_[nid1]->size() == 0) {
      data_arr_[nid1]->resize(nbins_);
    }
    if (data_arr_[nid2]->size() == 0) {
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

tree::GradStats BuildHist(const std::vector<GradientPair>& gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix& gmat,
                             GHistRow hist,
                             TlsType* hist_tls,
                             int32_t parent_nid,
                             const tree::TrainParam& param,
                             GHistRow sibling,
                             GHistRow parent,
                             int32_t this_nid,
                             int32_t another_nid,
                             const bool is_dense_layout) {
  static float prep = 0, histcomp = 0, reduce = 0;

  const size_t nthread = static_cast<size_t>(this->nthread_);

  const size_t* rid =  row_indices.begin;
  const size_t nrows = row_indices.Size();
  const uint32_t* index = gmat.index.data();
  const size_t* row_ptr =  gmat.row_ptr.data();
  const float* pgh = reinterpret_cast<const float*>(gpair.data());

  float* hist_data = reinterpret_cast<float*>(hist.data());

  const size_t n_elems_total = gmat.index.size();
  constexpr size_t elems_per_block = 40000;
  const size_t block_size = std::min(size_t(2048), (n_elems_total / elems_per_block > 0 ?
      n_elems_total / elems_per_block : nrows));
  constexpr size_t n_stack_size = 128;
  constexpr size_t cache_line_size = 64;
  constexpr size_t prefetch_offset = 10;

  size_t n_blocks = nrows/block_size;
  n_blocks += !!(nrows - n_blocks*block_size);

  const size_t nthread_to_process = std::min(nthread,  n_blocks);

  MemStackAllocator<uint32_t, n_stack_size> thread_init_alloc(nthread);
  MemStackAllocator<std::pair<tree::GradStats*, size_t>, n_stack_size> hist_local_alloc(nthread);
  MemStackAllocator<float, n_stack_size> thread_gh_sum_alloc(nthread*2);
  auto* p_thread_init = thread_init_alloc.Get();
  auto* p_hist_local = hist_local_alloc.Get();
  auto* p_thread_gh_sum = thread_gh_sum_alloc.Get();

  memset(p_thread_init, '\0', nthread*sizeof(p_thread_init[0]));
  memset(p_hist_local, '\0', nthread*sizeof(p_hist_local[0]));
  memset(p_thread_gh_sum, '\0', 2*nthread*sizeof(p_thread_gh_sum[0]));

  size_t no_prefetch_size = prefetch_offset + cache_line_size/sizeof(*rid);
  no_prefetch_size = no_prefetch_size > nrows ? nrows : no_prefetch_size;

  if (is_dense_layout) {
    ParallelFor(n_blocks, [&](size_t iblock) {
      dmlc::omp_uint tid = omp_get_thread_num();

      bool prev = p_thread_init[tid];
      if (!p_thread_init[tid]) {
        p_thread_init[tid] = true;
        p_hist_local[tid] = hist_tls->get(tid);
      }
      float* data_local_hist = ((nthread_to_process == 1) ? hist_data :
              reinterpret_cast<float*>(p_hist_local[tid].first));

      if (!prev)
        memset(data_local_hist, '\0', 2*nbins_*sizeof(float));

      const size_t istart = iblock*block_size;
      const size_t iend = (((iblock+1)*block_size > nrows) ? nrows : istart + block_size);

      const size_t n_features = row_ptr[rid[istart]+1] - row_ptr[rid[istart]];

      float gh_sum[2] = {0, 0};

      if (iend < nrows - no_prefetch_size) {
        for (size_t i = istart; i < iend; ++i) {
          const size_t icol_start = rid[i] * n_features;
          const size_t icol_start_prefetch = rid[i+prefetch_offset] * n_features;
          const size_t idx_gh = 2*rid[i];

          PREFETCH_READ_T0(pgh + 2*rid[i + prefetch_offset]);

          for (size_t j = icol_start_prefetch; j < icol_start_prefetch + n_features; j+=16)
            PREFETCH_READ_T0(index + j);

          gh_sum[0] += pgh[idx_gh];
          gh_sum[1] += pgh[idx_gh+1];

          for (size_t j = icol_start; j < icol_start + n_features; ++j) {
            const uint32_t idx_bin = 2*index[j];
            data_local_hist[idx_bin] += pgh[idx_gh];
            data_local_hist[idx_bin+1] += pgh[idx_gh+1];
          }
        }
      } else {
        for (size_t i = istart; i < iend; ++i) {
          const size_t icol_start = rid[i] * n_features;
          const size_t idx_gh = 2*rid[i];

          gh_sum[0] += pgh[idx_gh];
          gh_sum[1] += pgh[idx_gh+1];

          for (size_t j = icol_start; j < icol_start + n_features; ++j) {
            const uint32_t idx_bin      = 2*index[j];
            data_local_hist[idx_bin]   += pgh[idx_gh];
            data_local_hist[idx_bin+1] += pgh[idx_gh+1];
          }
        }
      }
      p_thread_gh_sum[tid*2] += gh_sum[0];
      p_thread_gh_sum[tid*2+1] += gh_sum[1];
    });
  } else {  // Sparse case
    ParallelFor(n_blocks, [&](size_t iblock) {
      dmlc::omp_uint tid = omp_get_thread_num();

      bool prev = p_thread_init[tid];
      if (!p_thread_init[tid]) {
        p_thread_init[tid] = true;
        p_hist_local[tid] = hist_tls->get(tid);
      }
      float* data_local_hist = ((nthread_to_process == 1) ? hist_data :
              reinterpret_cast<float*>(p_hist_local[tid].first));

      if (!prev) {
        memset(data_local_hist, '\0', 2*nbins_*sizeof(float));
      }

      const size_t istart = iblock*block_size;
      const size_t iend = (((iblock+1)*block_size > nrows) ? nrows : istart + block_size);

      float gh_sum[2] = {0, 0};

      if (iend < nrows - no_prefetch_size) {
        for (size_t i = istart; i < iend; ++i) {
          const size_t icol_start = row_ptr[rid[i]];
          const size_t icol_end = row_ptr[rid[i]+1];
          const size_t idx_gh = 2*rid[i];

          const size_t icol_start10 = row_ptr[rid[i+prefetch_offset]];
          const size_t icol_end10 = row_ptr[rid[i+prefetch_offset]+1];

          PREFETCH_READ_T0(pgh + 2*rid[i + prefetch_offset]);

          for (size_t j = icol_start10; j < icol_end10; j+=16) {
            PREFETCH_READ_T0(index + j);
          }

          gh_sum[0] += pgh[idx_gh];
          gh_sum[1] += pgh[idx_gh+1];

          for (size_t j = icol_start; j < icol_end; ++j) {
            const uint32_t idx_bin = 2*index[j];
            data_local_hist[idx_bin] += pgh[idx_gh];
            data_local_hist[idx_bin+1] += pgh[idx_gh+1];
          }
        }
      } else {
        for (size_t i = istart; i < iend; ++i) {
          const size_t icol_start = row_ptr[rid[i]];
          const size_t icol_end = row_ptr[rid[i]+1];
          const size_t idx_gh = 2*rid[i];

          gh_sum[0] += pgh[idx_gh];
          gh_sum[1] += pgh[idx_gh+1];

          for (size_t j = icol_start; j < icol_end; ++j) {
            const uint32_t idx_bin      = 2*index[j];
            data_local_hist[idx_bin]   += pgh[idx_gh];
            data_local_hist[idx_bin+1] += pgh[idx_gh+1];
          }
        }
      }
      p_thread_gh_sum[tid*2] += gh_sum[0];
      p_thread_gh_sum[tid*2+1] += gh_sum[1];
    });
  }


  {
    size_t n_worked_bins = 0;

    if (nthread_to_process == 0) {
      memset(hist_data, '\0', 2*nbins_*sizeof(float));
    } else if (nthread_to_process > 1) {
      for (size_t i = 0; i < nthread; ++i) {
        if (p_thread_init[i]) {
          std::swap(p_hist_local[n_worked_bins], p_hist_local[i]);
          n_worked_bins++;
        }
      }
    }

    const size_t size = (2*nbins_);
    const size_t block_size = 1024;  // aproximatly 1024 values per block
    size_t n_blocks = size/block_size + !!(size%block_size);

    ParallelFor(n_blocks, [&](size_t iblock) {
      const size_t ibegin = iblock*block_size;
      const size_t iend = (((iblock+1)*block_size > size) ? size : ibegin + block_size);

      if (nthread_to_process > 1) {
        memcpy(hist_data + ibegin, (reinterpret_cast<float*>(p_hist_local[0].first) + ibegin),
            sizeof(float)*(iend - ibegin));
        for (size_t i_bin_part = 1; i_bin_part < n_worked_bins; ++i_bin_part) {
          float* ptr = reinterpret_cast<float*>(p_hist_local[i_bin_part].first);
          for (size_t i = ibegin; i < iend; i++) {
            hist_data[i] += ptr[i];
          }
        }
      }

      if (another_nid > -1) {
        float* other = reinterpret_cast<float*>(sibling.data());
        float* par = reinterpret_cast<float*>(parent.data());

        for (size_t i = ibegin; i < iend; i++) {
          other[i] = par[i] - hist_data[i];
        }
      }
    });
  }

  for (uint32_t i = 0; i < nthread; i++) {
    if (p_hist_local[i].first) {
      hist_tls->release(p_hist_local[i]);
    }
  }

  tree::GradStats gh_sum;
  for (size_t i = 0; i < nthread; ++i) {
    gh_sum.sum_grad += p_thread_gh_sum[2 * i];
    gh_sum.sum_hess += p_thread_gh_sum[2 * i + 1];
  }
  return gh_sum;
}


void BuildBlockHist(const std::vector<GradientPair>& gpair,
                                  const RowSetCollection::Elem row_indices,
                                  const GHistIndexBlockMatrix& gmatb,
                                  GHistRow hist) {
  constexpr int kUnroll = 8;  // loop unrolling factor
  const size_t nblock = gmatb.GetNumBlock();
  const size_t nrows = row_indices.end - row_indices.begin;
  const size_t rest = nrows % kUnroll;

  ParallelFor(nblock, [&](size_t bid) {
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
  });
}

void SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent) {
  tree::GradStats* p_self = self.data();
  tree::GradStats* p_sibling = sibling.data();
  tree::GradStats* p_parent = parent.data();

  const size_t size = (2*nbins_);
  const size_t block_size = 1024;  // aproximatly 1024 values per block
  size_t n_blocks = size/block_size + !!(size%block_size);

  ParallelFor(n_blocks, [&](size_t iblock) {
    const size_t ibegin = iblock*block_size;
    const size_t iend = (((iblock+1)*block_size > size) ? size : ibegin + block_size);
    for (bst_omp_uint bin_id = ibegin; bin_id < iend; bin_id++) {
      p_self[bin_id].SetSubstract(p_parent[bin_id], p_sibling[bin_id]);
    }
  });
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


}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_UTIL_H_
