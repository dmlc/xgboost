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
#include "utils.cuh"
#include "../src/exact/gpu_builder.cuh"
#include "../src/exact/node.cuh"


namespace xgboost {
namespace tree {
namespace exact {

static const std::vector<int> smallColSizes = {0, 5, 0, 6, 4, 0, 0, 2, 0, 11,
                                               2, 9, 0, 5, 1, 0, 12, 3};

template <typename node_id_t>
void testSmallData() {
  GPUBuilder<node_id_t> builder;
  std::shared_ptr<DMatrix> dm =
      setupGPUBuilder<node_id_t>("plugin/updater_gpu/tests/data/small.sample.libsvm",
                              builder, 1);
  // data dimensions
  ASSERT_EQ(60, builder.nVals);
  ASSERT_EQ(15, builder.nRows);
  ASSERT_EQ(18, builder.nCols);
  ASSERT_TRUE(builder.allocated);
  // column counts
  int* tmpOff = new int[builder.nCols+1];
  updateHostPtr<int>(tmpOff, builder.colOffsets.data(), builder.nCols+1);
  for (int i = 0; i < 15; ++i) {
    EXPECT_EQ(smallColSizes[i], tmpOff[i+1]-tmpOff[i]);
  }
  float* tmpVal = new float[builder.nVals];
  updateHostPtr<float>(tmpVal, builder.vals.current(), builder.nVals);
  int* tmpInst = new int[builder.nVals];
  updateHostPtr<int>(tmpInst, builder.instIds.current(), builder.nVals);
  gpu_gpair* tmpGrad = new gpu_gpair[builder.nRows];
  updateHostPtr<gpu_gpair>(tmpGrad, builder.gradsInst.data(), builder.nRows);
  EXPECT_EQ(0, tmpInst[0]);
  EXPECT_FLOAT_EQ(1.f, tmpVal[0]);
  EXPECT_FLOAT_EQ(1.f+(float)(tmpInst[0]%10), get(0, tmpGrad, tmpInst).g);
  EXPECT_FLOAT_EQ(.5f+(float)(tmpInst[0]%10), get(0, tmpGrad, tmpInst).h);
  EXPECT_EQ(2, tmpInst[1]);
  EXPECT_FLOAT_EQ(1.f, tmpVal[1]);
  EXPECT_FLOAT_EQ(1.f+(float)(tmpInst[1]%10), get(1, tmpGrad, tmpInst).g);
  EXPECT_FLOAT_EQ(.5f+(float)(tmpInst[1]%10), get(1, tmpGrad, tmpInst).h);
  EXPECT_EQ(7, tmpInst[2]);
  EXPECT_FLOAT_EQ(1.f, tmpVal[2]);
  EXPECT_FLOAT_EQ(1.f+(float)(tmpInst[2]%10), get(2, tmpGrad, tmpInst).g);
  EXPECT_FLOAT_EQ(.5f+(float)(tmpInst[2]%10), get(2, tmpGrad, tmpInst).h);
  delete [] tmpGrad;
  delete [] tmpOff;
  delete [] tmpInst;
  delete [] tmpVal;
  int* colIds = new int[builder.nVals];
  updateHostPtr<int>(colIds, builder.colIds.data(), builder.nVals);
  std::vector<int> colSizeCopy(smallColSizes);
  int colIdxCurr = 0;
  for (int i = 0; i < builder.nVals; ++i) {
    while (colSizeCopy[colIdxCurr] == 0) {
      ++colIdxCurr;
    }
    --colSizeCopy[colIdxCurr];
    EXPECT_EQ(colIdxCurr, colIds[i]);
  }
  delete [] colIds;
}

TEST(CudaGPUBuilderTest, SetupOneTimeDataSmallInt16) {
  testSmallData<int16_t>();
}

TEST(CudaGPUBuilderTest, SetupOneTimeDataSmallInt32) {
  testSmallData<int>();
}

template <typename node_id_t>
void testLargeData() {
  GPUBuilder<node_id_t> builder;
  std::shared_ptr<DMatrix> dm =
      setupGPUBuilder<node_id_t>("plugin/updater_gpu/tests/data/sample.libsvm",
                              builder, 1);
  ASSERT_EQ(35442, builder.nVals);
  ASSERT_EQ(1611, builder.nRows);
  ASSERT_EQ(127, builder.nCols);
  ASSERT_TRUE(builder.allocated);
  int* tmpOff = new int[builder.nCols+1];
  updateHostPtr<int>(tmpOff, builder.colOffsets.data(), builder.nCols+1);
  EXPECT_EQ(0, tmpOff[1]-tmpOff[0]);   // 1st col
  EXPECT_EQ(83, tmpOff[2]-tmpOff[1]);  // 2nd col
  EXPECT_EQ(1, tmpOff[3]-tmpOff[2]);   // 3rd col
  float* tmpVal = new float[builder.nVals];
  updateHostPtr<float>(tmpVal, builder.vals.current(), builder.nVals);
  int* tmpInst = new int[builder.nVals];
  updateHostPtr<int>(tmpInst, builder.instIds.current(), builder.nVals);
  gpu_gpair* tmpGrad = new gpu_gpair[builder.nRows];
  updateHostPtr<gpu_gpair>(tmpGrad, builder.gradsInst.data(), builder.nRows);
  // the order of observations is messed up before the convertToCsc call!
  // hence, the instance IDs have been manually checked and put here.
  EXPECT_EQ(1164, tmpInst[0]);
  EXPECT_FLOAT_EQ(1.f, tmpVal[0]);
  EXPECT_FLOAT_EQ(1.f+(float)(tmpInst[0]%10), get(0, tmpGrad, tmpInst).g);
  EXPECT_FLOAT_EQ(.5f+(float)(tmpInst[0]%10), get(0, tmpGrad, tmpInst).h);
  EXPECT_EQ(1435, tmpInst[1]);
  EXPECT_FLOAT_EQ(1.f, tmpVal[1]);
  EXPECT_FLOAT_EQ(1.f+(float)(tmpInst[1]%10), get(1, tmpGrad, tmpInst).g);
  EXPECT_FLOAT_EQ(.5f+(float)(tmpInst[1]%10), get(1, tmpGrad, tmpInst).h);
  EXPECT_EQ(1421, tmpInst[2]);
  EXPECT_FLOAT_EQ(1.f, tmpVal[2]);
  EXPECT_FLOAT_EQ(1.f+(float)(tmpInst[2]%10), get(2, tmpGrad, tmpInst).g);
  EXPECT_FLOAT_EQ(.5f+(float)(tmpInst[2]%10), get(2, tmpGrad, tmpInst).h);
  delete [] tmpGrad;
  delete [] tmpOff;
  delete [] tmpInst;
  delete [] tmpVal;
}

TEST(CudaGPUBuilderTest, SetupOneTimeDataLargeInt16) {
  testLargeData<int16_t>();
}

TEST(CudaGPUBuilderTest, SetupOneTimeDataLargeInt32) {
  testLargeData<int>();
}

int getColId(int* offsets, int id, int nCols) {
  for (int i = 1; i <= nCols; ++i) {
    if (id < offsets[i]) {
      return (i-1);
    }
  }
  return -1;
}

template <typename node_id_t>
void testAllocate() {
  GPUBuilder<node_id_t> builder;
  std::shared_ptr<DMatrix> dm =
      setupGPUBuilder<node_id_t>("plugin/updater_gpu/tests/data/small.sample.libsvm",
                              builder, 1);
  ASSERT_EQ(3, builder.maxNodes);
  ASSERT_EQ(2, builder.maxLeaves);
  Node<node_id_t>* n = new Node<node_id_t>[builder.maxNodes];
  updateHostPtr<Node<node_id_t> >(n, builder.nodes.data(), builder.maxNodes);
  for (int i = 0; i < builder.maxNodes; ++i) {
    if (i == 0) {
      EXPECT_FALSE(n[i].isLeaf());
      EXPECT_FALSE(n[i].isUnused());
    } else {
      EXPECT_TRUE(n[i].isLeaf());
      EXPECT_FALSE(n[i].isUnused());
    }
  }
  gpu_gpair sum;
  sum.g = 0.f;
  sum.h = 0.f;
  for (int i = 0; i < builder.maxNodes; ++i) {
    if (!n[i].isUnused()) {
      sum += n[i].gradSum;
    }
  }
  // law of conservation of gradients! :)
  EXPECT_FLOAT_EQ(2.f*n[0].gradSum.g, sum.g);
  EXPECT_FLOAT_EQ(2.f*n[0].gradSum.h, sum.h);
  node_id_t* assigns = new node_id_t[builder.nVals];
  int* offsets = new int[builder.nCols+1];
  updateHostPtr<node_id_t>(assigns, builder.nodeAssigns.current(),
                           builder.nVals);
  updateHostPtr<int>(offsets, builder.colOffsets.data(), builder.nCols+1);
  for (int i = 0; i < builder.nVals; ++i) {
    EXPECT_EQ((node_id_t)0, assigns[i]);
  }
  delete [] n;
  delete [] assigns;
  delete [] offsets;
}

TEST(CudaGPUBuilderTest, AllocateNodeDataInt16) {
  testAllocate<int16_t>();
}

TEST(CudaGPUBuilderTest, AllocateNodeDataInt32) {
  testAllocate<int>();
}

template <typename node_id_t>
void assign(Node<node_id_t> *n, float g, float h, float sc, float wt,
            DefaultDirection d, float th, int c, int i) {
  n->gradSum.g = g;
  n->gradSum.h = h;
  n->score = sc;
  n->weight = wt;
  n->dir = d;
  n->threshold = th;
  n->colIdx = c;
  n->id = (node_id_t)i;
}

template <typename node_id_t>
void testMarkLeaves() {
  GPUBuilder<node_id_t> builder;
  std::shared_ptr<DMatrix> dm =
      setupGPUBuilder<node_id_t>("plugin/updater_gpu/tests/data/small.sample.libsvm",
                                             builder, 3);
  ASSERT_EQ(15, builder.maxNodes);
  ASSERT_EQ(8, builder.maxLeaves);
  Node<node_id_t>* hNodes = new Node<node_id_t>[builder.maxNodes];
  assign<node_id_t>(&hNodes[0], 2.f, 1.f, .75f, 0.5f, LeftDir, 0.25f, 0, 0);
  assign<node_id_t>(&hNodes[1], 2.f, 1.f, .75f, 0.5f, RightDir, 0.5f, 1, 1);
  assign<node_id_t>(&hNodes[2], 2.f, 1.f, .75f, 0.5f, LeftDir, 0.75f, 2, 2);
  assign<node_id_t>(&hNodes[3], 2.f, 1.f, .75f, 0.5f, RightDir, 1.f, 3, 3);
  assign<node_id_t>(&hNodes[4], 2.f, 1.f, .75f, 0.5f, LeftDir, 1.25f, 4, 4);
  hNodes[5] = Node<node_id_t>();
  assign<node_id_t>(&hNodes[6], 2.f, 1.f, .75f, 0.5f, LeftDir, 1.75f, 6, 6);
  hNodes[7] = Node<node_id_t>();
  hNodes[8] = Node<node_id_t>();
  hNodes[9] = Node<node_id_t>();
  hNodes[10] = Node<node_id_t>();
  hNodes[11] = Node<node_id_t>();
  hNodes[12] = Node<node_id_t>();
  hNodes[13] = Node<node_id_t>();
  hNodes[14] = Node<node_id_t>();
  updateDevicePtr<Node<node_id_t> >(builder.nodes.data(), hNodes, builder.maxNodes);
  builder.markLeaves();
  Node<node_id_t>* outNodes = new Node<node_id_t>[builder.maxNodes];
  updateHostPtr<Node<node_id_t> >(outNodes, builder.nodes.data(), builder.maxNodes);
  for (int i = 0; i < builder.maxNodes; ++i) {
    if ((i >= 7) || (i == 5)) {
      EXPECT_TRUE(outNodes[i].isUnused());
    } else {
      EXPECT_FALSE(outNodes[i].isUnused());
    }
  }
  for (int i = 0; i < builder.maxNodes; ++i) {
    if ((i == 3) || (i == 4) || (i == 6)) {
      EXPECT_TRUE(outNodes[i].isLeaf());
    } else {
      EXPECT_FALSE(outNodes[i].isLeaf());
    }
  }
  delete [] outNodes;
  delete [] hNodes;
}

TEST(CudaGPUBuilderTest, MarkLeavesInt16) {
  testMarkLeaves<int16_t>();
}

TEST(CudaGPUBuilderTest, MarkLeavesInt32) {
  testMarkLeaves<int>();
}

template <typename node_id_t>
void testDense2Sparse() {
  GPUBuilder<node_id_t> builder;
  std::shared_ptr<DMatrix> dm =
      setupGPUBuilder<node_id_t>("plugin/updater_gpu/tests/data/small.sample.libsvm",
                              builder, 3);
  ASSERT_EQ(15, builder.maxNodes);
  ASSERT_EQ(8, builder.maxLeaves);
  Node<node_id_t>* hNodes = new Node<node_id_t>[builder.maxNodes];
  assign<node_id_t>(&hNodes[0], 2.f, 1.f, .75f, 0.5f, LeftDir, 0.25f, 0, 0);
  assign<node_id_t>(&hNodes[1], 2.f, 1.f, .75f, 0.5f, RightDir, 0.5f, 1, 1);
  assign<node_id_t>(&hNodes[2], 2.f, 1.f, .75f, 0.5f, LeftDir, 0.75f, 2, 2);
  assign<node_id_t>(&hNodes[3], 2.f, 1.f, .75f, 0.5f, RightDir, 1.f, 3, 3);
  assign<node_id_t>(&hNodes[4], 2.f, 1.f, .75f, 0.5f, LeftDir, 1.25f, 4, 4);
  hNodes[5] = Node<node_id_t>();
  assign<node_id_t>(&hNodes[6], 2.f, 1.f, .75f, 0.5f, LeftDir, 1.75f, 6, 6);
  assign<node_id_t>(&hNodes[7], 2.f, 1.f, .75f, 0.5f, LeftDir, 1.75f, 7, 7);
  hNodes[8] = Node<node_id_t>();
  hNodes[9] = Node<node_id_t>();
  hNodes[10] = Node<node_id_t>();
  hNodes[11] = Node<node_id_t>();
  hNodes[12] = Node<node_id_t>();
  hNodes[13] = Node<node_id_t>();
  hNodes[14] = Node<node_id_t>();
  updateDevicePtr<Node<node_id_t> >(builder.nodes.data(), hNodes, builder.maxNodes);
  builder.markLeaves();
  RegTree tree;
  builder.dense2sparse(tree);
  EXPECT_EQ(9, tree.param.num_nodes);
  delete [] hNodes;
}

TEST(CudaGPUBuilderTest, Dense2SparseInt16) {
  testDense2Sparse<short int>();
}

TEST(CudaGPUBuilderTest, Dense2SparseInt32) {
  testDense2Sparse<int>();
}

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
