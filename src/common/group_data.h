/*!
 * Copyright 2014 by Contributors
 * \file group_data.h
 * \brief this file defines utils to group data by integer keys
 *     Input: given input sequence (key,value), (k1,v1), (k2,v2)
 *     Ouptupt: an array of values data = [v1,v2,v3 .. vn]
 *              and a group pointer ptr,
 *              data[ptr[k]:ptr[k+1]] contains values that corresponds to key k
 *
 * This can be used to construct CSR/CSC matrix from un-ordered input
 * The major algorithm is a two pass linear scan algorithm that requires two pass scan over the data
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_GROUP_DATA_H_
#define XGBOOST_COMMON_GROUP_DATA_H_

#include <vector>

namespace xgboost {
namespace common {
/*!
 * \brief multi-thread version of group builder
 * \tparam ValueType type of entries in the sparse matrix
 * \tparam SizeType type of the index range holder
 */
template<typename ValueType, typename SizeType = size_t>
struct ParallelGroupBuilder {
 public:
  // parallel group builder of data
  ParallelGroupBuilder(std::vector<SizeType> *p_rptr,
                       std::vector<ValueType> *p_data)
      : rptr_(*p_rptr), data_(*p_data), thread_rptr_(tmp_thread_rptr_) {
  }
  ParallelGroupBuilder(std::vector<SizeType> *p_rptr,
                       std::vector<ValueType> *p_data,
                       std::vector< std::vector<SizeType> > *p_thread_rptr)
      : rptr_(*p_rptr), data_(*p_data), thread_rptr_(*p_thread_rptr) {
  }

 public:
  /*!
   * \brief step 1: initialize the helper, with hint of number keys
   *                and thread used in the construction
   * \param nkeys number of keys in the matrix, can be smaller than expected
   * \param nthread number of thread that will be used in construction
   */
  inline void InitBudget(size_t nkeys, int nthread) {
    thread_rptr_.resize(nthread);
    for (size_t i = 0;  i < thread_rptr_.size(); ++i) {
      thread_rptr_[i].resize(nkeys);
      std::fill(thread_rptr_[i].begin(), thread_rptr_[i].end(), 0);
    }
  }
  /*!
   * \brief step 2: add budget to each key
   * \param key the key
   * \param threadid the id of thread that calls this function
   * \param nelem number of element budget add to this row
   */
  inline void AddBudget(size_t key, int threadid, SizeType nelem = 1) {
    std::vector<SizeType> &trptr = thread_rptr_[threadid];
    if (trptr.size() < key + 1) {
      trptr.resize(key + 1, 0);
    }
    trptr[key] += nelem;
  }
  /*! \brief step 3: initialize the necessary storage */
  inline void InitStorage() {
    // set rptr to correct size
    for (size_t tid = 0; tid < thread_rptr_.size(); ++tid) {
      if (rptr_.size() <= thread_rptr_[tid].size()) {
        rptr_.resize(thread_rptr_[tid].size() + 1);
      }
    }
    // initialize rptr to be beginning of each segment
    size_t start = 0;
    for (size_t i = 0; i + 1 < rptr_.size(); ++i) {
      for (size_t tid = 0; tid < thread_rptr_.size(); ++tid) {
        std::vector<SizeType> &trptr = thread_rptr_[tid];
        if (i < trptr.size()) {
          size_t ncnt = trptr[i];
          trptr[i] = start;
          start += ncnt;
        }
      }
      rptr_[i + 1] = start;
    }
    data_.resize(start);
  }
  /*!
   * \brief step 4: add data to the allocated space,
   *   the calls to this function should be exactly match previous call to AddBudget
   *
   * \param key the key of group.
   * \param value The value to be pushed to the group.
   * \param threadid the id of thread that calls this function
   */
  inline void Push(size_t key, ValueType value, int threadid) {
    SizeType &rp = thread_rptr_[threadid][key];
    data_[rp++] = value;
  }

 private:
  /*! \brief pointer to the beginning and end of each continuous key */
  std::vector<SizeType> &rptr_;
  /*! \brief index of nonzero entries in each row */
  std::vector<ValueType> &data_;
  /*! \brief thread local data structure */
  std::vector<std::vector<SizeType> > &thread_rptr_;
  /*! \brief local temp thread ptr, use this if not specified by the constructor */
  std::vector<std::vector<SizeType> > tmp_thread_rptr_;
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_GROUP_DATA_H_
