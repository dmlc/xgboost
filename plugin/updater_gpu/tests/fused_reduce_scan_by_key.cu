/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "gtest/gtest.h"
#include "../src/exact/fused_scan_reduce_by_key.cuh"
#include "../src/exact/node.cuh"
#include "utils.cuh"


namespace xgboost {
namespace tree {
namespace exact {

template <typename node_id_t>
class ReduceScanByKey: public Generator<node_id_t> {
 public:
  ReduceScanByKey(int nc, int nr, int nk, const std::string& tName):
    Generator<node_id_t>(nc, nr, nk, tName),
    hSums(nullptr), dSums(nullptr), hScans(nullptr), dScans(nullptr),
    outSize(this->size), nSegments(this->nKeys*this->nCols),
    hOffsets(nullptr), dOffsets(nullptr) {
    hSums = new gpu_gpair[nSegments];
    allocateOnGpu<gpu_gpair>(dSums, nSegments);
    hScans = new gpu_gpair[outSize];
    allocateOnGpu<gpu_gpair>(dScans, outSize);
    gpu_gpair* buckets = new gpu_gpair[nSegments];
    for (int i=0;i<nSegments;i++) {
      buckets[i] = gpu_gpair();
    }
    for (int i=0;i<nSegments;i++) {
      hSums[i] = gpu_gpair();
    }
    for (size_t i=0;i<this->size;i++) {
      if (this->hKeys[i] >= 0 && this->hKeys[i] < nSegments) {
        node_id_t key = abs2uniqKey<node_id_t>(i, this->hKeys,
                                               this->hColIds, 0,
                                               this->nKeys);
        hSums[key] += this->hVals[i];
      }
    }
    for (int i=0;i<this->size;++i) {
      node_id_t key = abs2uniqKey<node_id_t>(i, this->hKeys,
                                             this->hColIds, 0,
                                             this->nKeys);
      hScans[i] = buckets[key];
      buckets[key] += this->hVals[i];
    }
    // it's a dense matrix that we are currently looking at, so offsets
    // are nicely aligned! (need not be the case in real datasets)
    hOffsets = new int[this->nCols];
    size_t off = 0;
    for (int i=0;i<this->nCols;++i,off+=this->nRows) {
      hOffsets[i] = off;
    }
    allocateAndUpdateOnGpu<int>(dOffsets, hOffsets, this->nCols);
  }

  ~ReduceScanByKey() {
    delete [] hScans;
    delete [] hSums;
    delete [] hOffsets;
    dh::safe_cuda(cudaFree(dScans));
    dh::safe_cuda(cudaFree(dSums));
    dh::safe_cuda(cudaFree(dOffsets));
  }

  void run() {
    gpu_gpair* tmpScans;
    int* tmpKeys;
    int tmpSize = scanTempBufferSize(this->size);
    allocateOnGpu<gpu_gpair>(tmpScans, tmpSize);
    allocateOnGpu<int>(tmpKeys, tmpSize);
    TIMEIT(reduceScanByKey<node_id_t>
           (dSums, dScans, this->dVals, this->dInstIds, this->dKeys,
            this->size, this->nKeys, this->nCols, tmpScans, tmpKeys,
            this->dColIds, 0),
           this->testName);
    dh::safe_cuda(cudaFree(tmpScans));
    dh::safe_cuda(cudaFree(tmpKeys));
    this->compare(hSums, dSums, nSegments);
    this->compare(hScans, dScans, outSize);     
  }

 private:
  gpu_gpair* hSums;
  gpu_gpair* dSums;
  gpu_gpair* hScans;
  gpu_gpair* dScans;
  int outSize;
  int nSegments;
  int* hOffsets;
  int* dOffsets;
};

TEST(ReduceScanByKey, testInt16) {
  ReduceScanByKey<short int>(32, 512, 32, "ReduceScanByKey").run();
}

TEST(ReduceScanByKey, testInt32) {
  ReduceScanByKey<int>(32, 512, 32, "ReduceScanByKey").run();
}

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
