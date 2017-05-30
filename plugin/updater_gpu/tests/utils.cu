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
#include "utils.cuh"


namespace xgboost {
namespace tree {
namespace exact {

std::shared_ptr<DMatrix> generateData(const std::string& file) {
  std::shared_ptr<DMatrix> data(DMatrix::Load(file, false, false, "libsvm"));
  return data;
}

std::shared_ptr<DMatrix> preparePluginInputs(const std::string& file,
                                             std::vector<bst_gpair>& gpair) {
  std::shared_ptr<DMatrix> dm = generateData(file);
  gpair.reserve(dm->info().num_row);
  for (int i=0;i<dm->info().num_row;++i) {
    gpair.push_back(bst_gpair(1.f+(float)(i%10), 0.5f+(float)(i%10)));
  }
  return dm;
}

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
