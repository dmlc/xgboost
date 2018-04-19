/*!
 * Copyright 2014 by Contributors
 * \file bitmap.h
 * \brief a simple implement of bitmap
 *  NOTE: bitmap is only threadsafe per word access, remember this when using bitmap
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_BITMAP_H_
#define XGBOOST_COMMON_BITMAP_H_

#include <dmlc/omp.h>
#include <vector>

namespace xgboost {
namespace common {
/*! \brief bit map that contains set of bit indicators */
struct BitMap {
  /*! \brief internal data structure */
  std::vector<uint32_t> data;
  /*!
   * \brief resize the bitmap to be certain size
   * \param size the size of bitmap
   */
  inline void Resize(size_t size) {
    data.resize((size + 31U) >> 5, 0);
  }
  /*!
   * \brief query the i-th position of bitmap
   * \param i the position in
   */
  inline bool Get(size_t i) const {
    return (data[i >> 5] >> (i & 31U)) & 1U;
  }
  /*!
   * \brief set i-th position to true
   * \param i position index
   */
  inline void SetTrue(size_t i) {
    data[i >> 5] |= (1 << (i & 31U));
  }
  /*! \brief initialize the value of bit map from vector of bool*/
  inline void InitFromBool(const std::vector<int>& vec) {
    this->Resize(vec.size());
    // parallel over the full cases
    auto nsize = static_cast<bst_omp_uint>(vec.size() / 32);
    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < nsize; ++i) {
      uint32_t res = 0;
      for (int k = 0; k < 32; ++k) {
        int bit = vec[(i << 5) | k];
        res |= (bit << k);
      }
      data[i] = res;
    }
    if (nsize != vec.size()) data.back() = 0;
    for (size_t i = nsize; i < vec.size(); ++i) {
      if (vec[i]) this->SetTrue(i);
    }
  }
  /*! \brief clear the bitmap, set all places to false */
  inline void Clear() {
    std::fill(data.begin(), data.end(), 0U);
  }
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_BITMAP_H_
