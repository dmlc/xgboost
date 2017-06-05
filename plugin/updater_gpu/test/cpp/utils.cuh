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
#pragma once

#include <random>
#include "../../src/exact/gradients.cuh"
#include <memory>
#include <string>
#include <xgboost/data.h>
#include "gtest/gtest.h"
#include "../../src/exact/gpu_builder.cuh"
#include "../../src/device_helpers.cuh"
#include <vector>
#include <stdlib.h>


namespace xgboost {
namespace tree {
namespace exact {


template <typename T>
inline void allocateOnGpu(T*& arr, size_t nElems) {
  dh::safe_cuda(cudaMalloc((void**)&arr, sizeof(T)*nElems));
}

template <typename T>
inline void updateDevicePtr(T* dArr, const T* hArr, size_t nElems) {
  dh::safe_cuda(cudaMemcpy(dArr, hArr, sizeof(T)*nElems, cudaMemcpyHostToDevice));
}

template <typename T>
inline void updateHostPtr(T* hArr, const T* dArr, size_t nElems) {
  dh::safe_cuda(cudaMemcpy(hArr, dArr, sizeof(T)*nElems, cudaMemcpyDeviceToHost));
}

template <typename T>
inline void allocateAndUpdateOnGpu(T*& dArr, const T* hArr, size_t nElems) {
  allocateOnGpu<T>(dArr, nElems);
  updateDevicePtr<T>(dArr, hArr, nElems);
}

static const float Thresh = 0.005f;
static const float SuperSmall = 0.001f;
static const float SuperSmallThresh = 0.00001f;

// lets assume dense matrix for simplicity
template <typename T>
class Generator {
 public:
  Generator(int nc, int nr, int nk, const std::string& tName):
    nCols(nc), nRows(nr), nKeys(nk), size(nc*nr), hKeys(nullptr),
    dKeys(nullptr), hVals(nullptr), dVals(nullptr), testName(tName),
    dColIds(nullptr), hColIds(nullptr), dInstIds(nullptr),
    hInstIds(nullptr) {
    generateKeys();
    generateVals();
    // to simulate the same sorted key-value pairs in the main code
    // don't need it as generateKeys always generates in sorted order!
    //sortKeyValues();
    evalColIds();
    evalInstIds();
  }

  virtual ~Generator() {
    delete [] hKeys;
    delete [] hVals;
    delete [] hColIds;
    delete [] hInstIds;
    dh::safe_cuda(cudaFree(dColIds));
    dh::safe_cuda(cudaFree(dKeys));
    dh::safe_cuda(cudaFree(dVals));
    dh::safe_cuda(cudaFree(dInstIds));
  }

  virtual void run() = 0;

protected:
  int nCols;
  int nRows;
  int nKeys;
  int size;
  T* hKeys;
  T* dKeys;
  gpu_gpair* hVals;
  gpu_gpair* dVals;
  std::string testName;
  int* dColIds;
  int* hColIds;
  int* dInstIds;
  int* hInstIds;

  void evalColIds() {
    hColIds = new int[size];
    for (int i=0;i<size;++i) {
      hColIds[i] = i / nRows;
    }
    allocateAndUpdateOnGpu<int>(dColIds, hColIds, size);
  }

  void evalInstIds() {
    hInstIds = new int[size];
    for (int i=0;i<size;++i) {
      hInstIds[i] = i;
    }
    allocateAndUpdateOnGpu<int>(dInstIds, hInstIds, size);
  }

  float diffRatio(float a, float b, bool& isSmall) {
    isSmall = true;
    if (a == 0.f) return fabs(b);
    else if (b == 0.f) return fabs(a);
    else if ((fabs(a) < SuperSmall) && (fabs(b) < SuperSmall)) {
      return fabs(a - b);
    }
    else {
      isSmall = false;
      return fabs((a < b)? (b - a)/b : (a - b)/a);
    }
  }

  void compare(gpu_gpair* exp, gpu_gpair* dAct, size_t len) {
    gpu_gpair* act = new gpu_gpair[len];
    updateHostPtr<gpu_gpair>(act, dAct, len);
    for (size_t i=0;i<len;++i) {
      bool isSmall;
      float ratioG = diffRatio(exp[i].g, act[i].g, isSmall);
      float ratioH = diffRatio(exp[i].h, act[i].h, isSmall);
      float thresh = isSmall? SuperSmallThresh : Thresh;
      if ((ratioG >= Thresh) || (ratioH >= Thresh)) {
        printf("(exp) %f %f -> (act) %f %f : rG=%f rH=%f th=%f @%lu\n",
               exp[i].g, exp[i].h, act[i].g, act[i].h, ratioG, ratioH,
               thresh, i);
      }
      ASSERT_TRUE(ratioG < thresh);
      ASSERT_TRUE(ratioH < thresh);
    }
    delete [] act;
  }

  void generateKeys() {
    hKeys = new T[size];
    T currKey = 0;
    for (int i=0;i<size;++i) {
      if (i % nRows == 0) { // start fresh for a new column
        currKey = 0;
      }
      hKeys[i] = currKey;
      float val = randVal();
      if ((val > 0.8f) && (currKey < nKeys-1)) {
        ++currKey;
      }
    }
    allocateAndUpdateOnGpu<T>(dKeys, hKeys, size);
  }

  void generateVals() {
    hVals = new gpu_gpair[size];
    for (size_t i=0;i<size;++i) {
      hVals[i].g = randVal(-1.f, 1.f);
      hVals[i].h = randVal(-1.f, 1.f);
    }
    allocateAndUpdateOnGpu<gpu_gpair>(dVals, hVals, size);
  }

  void sortKeyValues() {
    char* storage = nullptr;
    size_t tmpSize;
    dh::safe_cuda(cub::DeviceRadixSort::SortPairs(NULL, tmpSize, dKeys, dKeys,
                                               dVals, dVals, size));
    allocateOnGpu<char>(storage, tmpSize);
    void* tmpStorage = static_cast<void*>(storage);
    dh::safe_cuda(cub::DeviceRadixSort::SortPairs(tmpStorage, tmpSize, dKeys,
                                               dKeys, dVals, dVals, size));
    dh::safe_cuda(cudaFree(storage));
    updateHostPtr<gpu_gpair>(hVals, dVals, size);
    updateHostPtr<T>(hKeys, dKeys, size);
  }

  float randVal() const {
    float val = rand() * 1.f / RAND_MAX;
    return val;
  }

  float randVal(float min, float max) const {
    float val = randVal();
    val = (val * (max - min)) - min;
    return val;
  }
};


std::shared_ptr<DMatrix> generateData(const std::string& file);

std::shared_ptr<DMatrix> preparePluginInputs(const std::string& file,
                                             std::vector<bst_gpair> *gpair);

template <typename node_id_t>
std::shared_ptr<DMatrix> setupGPUBuilder(const std::string& file,
                                         GPUBuilder<node_id_t> &builder,
                                         int max_depth=1) {
  std::vector<bst_gpair> gpair;
  std::shared_ptr<DMatrix> dm = preparePluginInputs(file, &gpair);
  TrainParam p;
  RegTree tree;
  p.min_split_loss = 0.f;
  p.max_depth = max_depth;
  p.min_child_weight = 0.f;
  p.reg_alpha = 0.f;
  p.reg_lambda = 1.f;
  p.max_delta_step = 0.f;
  builder.Init(p);
  builder.Update(gpair, dm.get(), &tree);
  return dm;
}

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
