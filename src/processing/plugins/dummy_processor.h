/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#pragma once
#include <string>
#include <cstring>
#include <vector>
#include <map>
#include "../processor.h"

class DummyProcessor: public processing::Processor {
 private:
    bool active_ = false;
    const std::map<std::string, std::string> *params_{nullptr};
    std::vector<double> *gh_pairs_{nullptr};
    std::vector<uint32_t> cuts_;
    std::vector<int> slots_;

 public:
    void Initialize(bool active, std::map<std::string, std::string> params) override {
        this->active_ = active;
        this->params_ = &params;
    }

    void Shutdown() override {
        this->gh_pairs_ = nullptr;
        this->cuts_.clear();
        this->slots_.clear();
    }

    void FreeBuffer(void *buffer) override {
        free(buffer);
    }

    void* ProcessGHPairs(size_t &size, std::vector<double>& pairs) override;

    void* HandleGHPairs(size_t &size, void *buffer, size_t buf_size) override;

    void InitAggregationContext(const std::vector<uint32_t> &cuts, std::vector<int> &slots) override {
        std::cout << "InitAggregationContext called with cuts size: " << cuts.size()-1 <<
           " number of slot: " << slots.size() << std::endl;
        this->cuts_ = cuts;
        if (this->slots_.empty()) {
            this->slots_ = slots;
        } else {
            std::cout << "Multiple calls to InitAggregationContext" << std::endl;
        }
    }

    void *ProcessAggregation(size_t &size, std::map<int, std::vector<int>> nodes) override;

    std::vector<double> HandleAggregation(void *buffer, size_t buf_size) override;
};
