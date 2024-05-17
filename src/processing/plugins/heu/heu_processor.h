// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "heu/library/numpy/numpy.h"

#include "processing/processor.h"

namespace processing {

class HeuProcessor : public Processor {
 public:
  void Initialize(bool active,
                  std::map<std::string, std::string> params) override;

  void Shutdown() override;

  void FreeBuffer(void *buffer) override;

  void *ProcessGHPairs(size_t *size, const std::vector<double> &pairs) override;

  void *HandleGHPairs(size_t *size, void *buffer, size_t buf_size) override;

  void InitAggregationContext(const std::vector<uint32_t> &cuts,
                              const std::vector<int> &slots) override;

  void *ProcessAggregation(size_t *size,
                           std::map<int, std::vector<int>> nodes) override;

  std::vector<double> HandleAggregation(void *buffer, size_t buf_size) override;

 private:
  bool active_ = false;
  int64_t scale_ = 0;
  std::vector<uint32_t> cuts_;
  std::vector<int> slots_;
  std::unique_ptr<heu::lib::numpy::CMatrix> gh_ = nullptr;
  std::unique_ptr<heu::lib::numpy::HeKit> he_kit_ = nullptr;
  std::unique_ptr<heu::lib::numpy::DestinationHeKit> dest_he_kit_ = nullptr;
};

}  // namespace processing
