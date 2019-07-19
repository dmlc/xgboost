/*!
 * Copyright 2019 by XGBoost Contributors
 * \file data.cuh
 * \brief An extension for the data interface to support foreign columnar data buffers
    This file adds the necessary functions to fill the meta information for the columnar buffers
 * \author Andrey Adinets
 * \author Matthew Jones
 */
#include <xgboost/data.h>
#include <xgboost/logging.h>

#include "../common/device_helpers.cuh"
#include "../common/host_device_vector.h"

namespace xgboost {

__global__ void ReadColumn(ForeignColumn * col,
                             foreign_size_type n_cols,
                             void * data) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  foreign_size_type n_rows = col->size;
  if (n_rows <= tid) {
    return;
  } else {
    data[n_cols * tid] = float(col->data[tid]);
  }
}

void SetInfoFromForeignColumns(MetaInfo * info,
                               const char * key,
                               ForeignColumn ** cols,
                               foreign_size_type n_cols) {
  CHECK_GT(n_cols, 0);
  foreign_size_type n_rows = cols[0]->size;
  for (foreign_size_type i = 0; i < n_cols; ++i) {
    CHECK_EQ(n_rows, cols[i]->size) << "all foreign columns must be the same size";
    CHECK_EQ(cols[i]->null_count, 0) << "all labels and weights must be valid";
  }
  HostDeviceVector<bst_float> * field = nullptr;
  if(!strcmp(key, "label")) {
    field = &info->labels_;
  } else if (!strcmp(key, "weight")) {
    CHECK_EQ(n_cols, 1) << "only one foreign column permitted for weights";
    field = &info->weights_;
  } else {
    LOG(WARNING) << key << ": invalid key value for MetaInfo field";
  }

  GPUSet devices = GPUSet::Range(0, 1);
  field->Reshard(GPUDistribution::Granular(devices, n_cols));
  field->Resize(n_cols * n_rows);
  bst_float * data = field->DevicePointer(0);

  int threads = 1024;
  int blocks = (n_rows + threads - 1) / threads;

  for (foreign_size_type i = 0; i < n_cols; ++i) {
    ReadColumn <<<threads, blocks>>> (cols[i], n_cols, data + i);
    dh::safe_cuda(cudaGetLastError());
  }
}
}  // namespace xgboost