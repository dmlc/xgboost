#ifndef XGBOOST_DATA_COLUMNAR_H_
#define XGBOOST_DATA_COLUMNAR_H_

#include <atomic>
#include <cinttypes>
#include "../common/span.h"

#if defined(XGBOOST_USE_CUDA)
#include "../common/bitfield.cuh"
#else
#include <bitset>
#endif  // defined(XGBOOST_USE_CUDA)

namespace xgboost {

typedef unsigned char foreign_valid_type;
typedef int32_t foreign_size_type;

struct ForeignColumn {
  common::Span<float>  data;
  BitField valid;
  foreign_size_type size;
  foreign_size_type null_count;
};

}      // namespace xgboost
#endif  // XGBOOST_DATA_COLUMNAR_H_
