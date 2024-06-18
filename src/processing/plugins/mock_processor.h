/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#pragma once
#include <string>
#include <vector>
#include <map>
#include "../processor.h"

//  Data type definition
const int64_t kDataTypeGHPairs = 1;
const int64_t kDataTypeHisto = 2;
const int64_t kDataTypeAggregatedHisto = 3;

class MockProcessor: public processing::Processor {
 private:
  bool active_ = false;
  const std::map<std::string, std::string> *params_{nullptr};
  std::vector<double> gh_pairs_;
  std::vector<uint32_t> cuts_;
  std::vector<int> slots_;

 public:
  void Initialize(bool active, std::map<std::string, std::string> params) override {
    this->active_ = active;
    this->params_ = &params;
  }

  void Shutdown() override {
    this->gh_pairs_.clear();
    this->cuts_.clear();
    this->slots_.clear();
  }

  void FreeBuffer(void *buffer) override {
    free(buffer);
  }

  void* ProcessGHPairs(size_t *size, const std::vector<double>& pairs) override;

  void* HandleGHPairs(size_t *size, void *buffer, size_t buf_size) override;

  void InitAggregationContext(const std::vector<uint32_t> &cuts,
                               const std::vector<int> &slots) override {
    this->cuts_ = cuts;
    if (this->slots_.empty()) {
      this->slots_ = slots;
    }
  }

  void *ProcessAggregation(size_t *size, std::map<int, std::vector<int>> nodes) override;

  std::vector<double> HandleAggregation(void *buffer, size_t buf_size) override;

  void *ProcessHistograms(size_t *size, const std::vector<double>& histograms) override;

  std::vector<double> HandleHistograms(void *buffer, size_t buf_size) override;
};
