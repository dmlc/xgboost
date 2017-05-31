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
#include "../src/exact/argmax_by_key.cuh"
#include "../src/exact/gradients.cuh"
#include "../src/exact/node.cuh"
#include "../src/exact/loss_functions.cuh"
#include "utils.cuh"


namespace xgboost {
namespace tree {
namespace exact {

TEST(ArgMaxByKey, maxSplit) {
  Split a, b, out;
  a.score = 2.f;
  a.index = 3;
  b.score = 3.f;
  b.index = 4;
  out = maxSplit(a, b);
  EXPECT_FLOAT_EQ(out.score, b.score);
  EXPECT_EQ(out.index, b.index);
  b.score = 2.f;
  b.index = 4;
  out = maxSplit(a, b);
  EXPECT_FLOAT_EQ(out.score, a.score);
  EXPECT_EQ(out.index, a.index);
  b.score = 2.f;
  b.index = 2;
  out = maxSplit(a, b);
  EXPECT_FLOAT_EQ(out.score, a.score);
  EXPECT_EQ(out.index, b.index);
  b.score = 1.f;
  b.index = 1;
  out = maxSplit(a, b);
  EXPECT_FLOAT_EQ(out.score, a.score);
  EXPECT_EQ(out.index, a.index);
}

template <typename node_id_t>
void argMaxTest(ArgMaxByKeyAlgo algo) {
  const int nVals = 1024;
  const int level = 0;
  const int nKeys = 1 << level;
  gpu_gpair* scans = new gpu_gpair[nVals];
  float* vals = new float[nVals];
  int* colIds = new int[nVals];
  scans[0] = gpu_gpair();
  vals[0] = 0.f;
  colIds[0] = 0;
  for (int i = 1; i < nVals; ++i) {
    scans[i].g = scans[i-1].g + (0.1f * 2.f);
    scans[i].h = scans[i-1].h + (0.1f * 2.f);
    vals[i] = static_cast<float>(i) * 0.1f;
    colIds[i] = 0;
  }
  float* dVals;
  allocateAndUpdateOnGpu<float>(dVals, vals, nVals);
  gpu_gpair* dScans;
  allocateAndUpdateOnGpu<gpu_gpair>(dScans, scans, nVals);
  gpu_gpair* sums = new gpu_gpair[nKeys];
  sums[0].g = sums[0].h = (0.1f * 2.f * nVals);
  gpu_gpair* dSums;
  allocateAndUpdateOnGpu<gpu_gpair>(dSums, sums, nKeys);
  int* dColIds;
  allocateAndUpdateOnGpu<int>(dColIds, colIds, nVals);
  Split* splits = new Split[nKeys];
  Split* dSplits;
  allocateOnGpu<Split>(dSplits, nKeys);
  node_id_t* nodeAssigns = new node_id_t[nVals];
  memset(nodeAssigns, 0, sizeof(node_id_t)*nVals);
  node_id_t* dNodeAssigns;
  allocateAndUpdateOnGpu<node_id_t>(dNodeAssigns, nodeAssigns, nVals);
  Node<node_id_t>* nodes = new Node<node_id_t>[nKeys];
  nodes[0].gradSum = sums[0];
  nodes[0].id = 0;
  TrainParam param;
  param.min_child_weight = 0.0f;
  param.reg_alpha = 0.f;
  param.reg_lambda = 2.f;
  param.max_delta_step = 0.f;
  nodes[0].score = CalcGain(param, sums[0].g, sums[0].h);
  Node<node_id_t>* dNodes;
  allocateAndUpdateOnGpu<Node<node_id_t> >(dNodes, nodes, nKeys);
  argMaxByKey<node_id_t>(dSplits, dScans, dSums, dVals, dColIds, dNodeAssigns,
                         dNodes, nKeys, 0, nVals, param, algo);
  updateHostPtr<Split>(splits, dSplits, nKeys);
  EXPECT_FLOAT_EQ(0.f, splits->score);
  EXPECT_EQ(0, splits->index);
  dh::safe_cuda(cudaFree(dNodeAssigns));
  delete [] nodeAssigns;
  dh::safe_cuda(cudaFree(dSplits));
  delete [] splits;
  dh::safe_cuda(cudaFree(dColIds));
  delete [] colIds;
  dh::safe_cuda(cudaFree(dSums));
  delete [] sums;
  dh::safe_cuda(cudaFree(dVals));
  delete [] vals;
  dh::safe_cuda(cudaFree(dScans));
  delete [] scans;
}

TEST(ArgMaxByKey, testOneColGmem) {
  argMaxTest<int16_t>(ABK_GMEM);
}

TEST(ArgMaxByKey, testOneColSmem) {
  argMaxTest<int16_t>(ABK_SMEM);
}

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
