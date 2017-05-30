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
#include "../src/exact/node.cuh"


namespace xgboost {
namespace tree {
namespace exact {

TEST(Split, Test) {
  Split s;
  EXPECT_FALSE(s.isSplittable(0.5f));
  s.score = 1.f;
  EXPECT_FALSE(s.isSplittable(0.5f));
  s.index = 2;
  EXPECT_TRUE(s.isSplittable(0.5f));
  EXPECT_FALSE(s.isSplittable(1.5f));
}

TEST(Node, Test) {
  Node<short int> n;
  EXPECT_TRUE(n.isUnused());
  EXPECT_FALSE(n.isLeaf());
  EXPECT_TRUE(n.isDefaultLeft());
  n.dir = RightDir;
  EXPECT_TRUE(n.isUnused());
  EXPECT_FALSE(n.isLeaf());
  EXPECT_FALSE(n.isDefaultLeft());
  n.id = 123;
  EXPECT_FALSE(n.isUnused());
  EXPECT_TRUE(n.isLeaf());
  EXPECT_FALSE(n.isDefaultLeft());
  n.score = 0.5f;
  EXPECT_FALSE(n.isUnused());
  EXPECT_FALSE(n.isLeaf());
  EXPECT_FALSE(n.isDefaultLeft());
}

TEST(Segment, Test) {
  Segment s;
  EXPECT_FALSE(s.isValid());
  s.start = 2;
  EXPECT_FALSE(s.isValid());
  s.end = 4;
  EXPECT_TRUE(s.isValid());
}

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
