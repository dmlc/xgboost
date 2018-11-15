/*!
 * Copyright 2018 by xgboost contributors
 */

#ifdef XGBOOST_USE_GDF
#include <gdf/gdf.h>
#include <xgboost/data.h>
#include <xgboost/logging.h>

#include "../common/device_helpers.cuh"
#include "../common/host_device_vector.h"
#include "./gdf.cuh"

namespace xgboost {

using namespace data;

__global__ void unpack_gdf_column_k
  (float* data, size_t n_rows, size_t n_cols, gdf_column col) {
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rows)
    return;
  data[n_cols * i] = convert_data_element(col.data, i, col.dtype);
}

void MetaInfo::SetInfoGDF(const char* key, gdf_column** cols, size_t n_cols) {
  CHECK_GT(n_cols, 0);
  size_t n_rows = cols[0]->size;
  for (size_t i = 0; i < n_cols; ++i) {
    CHECK_EQ(cols[i]->null_count, 0) << "all labels and weights must be valid";
    CHECK_EQ(cols[i]->size, n_rows) << "all GDF columns must be of the same size";
  }
  HostDeviceVector<bst_float>* field = nullptr;
  if (!strcmp(key, "label")) {
    field = &labels_;
  } else if (!strcmp(key, "weight")) {
    field = &weights_;
    CHECK_EQ(n_cols, 1) << "only one GDF column allowed for weights";
  } else {
    LOG(WARNING) << key << ": invalid key value for MetaInfo field";
    return;
  }
  // TODO(canonizer): use the same devices as elsewhere in xgboost
  GPUSet devices = GPUSet::Range(0, 1);
  field->Reshard(GPUDistribution::Granular(devices, n_cols));
  field->Resize(n_cols * n_rows);
  bst_float* data = field->DevicePointer(0);
  for (size_t i = 0; i < n_cols; ++i) {
    int block = 256;
    unpack_gdf_column_k<<<dh::DivRoundUp(n_rows, block), block>>>
      (data + i, n_rows, n_cols, *cols[i]);
    dh::safe_cuda(cudaGetLastError());
  }
}
  
}  // namespace xgboost
#endif
