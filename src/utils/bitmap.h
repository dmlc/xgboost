#ifndef XGBOOST_UTILS_BITMAP_H_
#define XGBOOST_UTILS_BITMAP_H_
/*!
 * \file bitmap.h
 * \brief a simple implement of bitmap
 * \author Tianqi Chen
 */
#include <vector>
#include "./utils.h"

namespace xgboost {
namespace utils {
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
  /*! \brief clear the bitmap, set all places to false */
  inline void Clear(void) {
    std::fill(data.begin(), data.end(), 0U);
  }
};
}  // namespace utils
}  // namespace xgboost
#endif
