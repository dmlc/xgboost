/*!
 * Copyright 2018 by xgboost contributors
 */

#include <cudf/types.h>

namespace xgboost {
namespace data {
  
/**
 * Convert the data element into a common format
 */
__device__ inline float ConvertDataElement(void* data, int tid, gdf_dtype dtype) {
  switch(dtype) {
    case gdf_dtype::GDF_INT8: {
      int8_t * d = (int8_t*)data;
      return float(d[tid]]);
    }
    case gdf_dtype::GDF_INT16: {
      int16_t * d = (int16_t*)data;
      return float(d[tid]]);
    }
    case gdf_dtype::GDF_INT32: {
      int32_t * d = (int32_t*)data;
      return float(d[tid]]);
    }
    case gdf_dtype::GDF_INT64: {
      int64_t * d = (int64_t*)data;
      return float(d[tid]]);
    }
    case gdf_dtype::GDF_FLOAT32: {
      float * d = (float *)data;
      return float(d[tid]]);
    }
    case gdf_dtype::GDF_FLOAT64: {
      double * d = (double *)data;
      return float(d[tid]]);
    }
  }
  return nanf(nullptr);
}

}  // namespace data
}  // namespace xgboost
