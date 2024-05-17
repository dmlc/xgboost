/**
 * Copyright 2014-2024 by XGBoost Contributors
 */

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
