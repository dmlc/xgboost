/*!
 * Copyright 2018 by xgboost contributors
 */

#include <cudf.h>

namespace xgboost {
namespace data {
  
/**
 * Convert the data element into a common format
 */
__device__ inline float ConvertDataElement(void* data, int tid, gdf_dtype dtype) {
  switch(dtype) {
    case gdf_dtype::GDF_INT8: {
      int8_t* a = (int8_t*)data;
      return float(a[tid]);
    }
    case gdf_dtype::GDF_INT16: {
      int16_t* a = (int16_t*)data;
      return float(a[tid]);
    }
    case gdf_dtype::GDF_INT32: {
      int32_t* a = (int32_t*)data;
      return float(a[tid]);
    }
    case gdf_dtype::GDF_INT64: {
      int64_t* a = (int64_t*)data;
      return float(a[tid]);
    }
    case gdf_dtype::GDF_FLOAT32: {
      float *a = (float *)data;
      return float(a[tid]);
    }
    case gdf_dtype::GDF_FLOAT64: {
      double *a = (double *)data;
      return float(a[tid]);
    }
  }
  return nanf(nullptr);
}

}  // namespace data
}  // namespace xgboost
