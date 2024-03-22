/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#pragma once
#include <string>
#include <vector>
#include <map>
#include "../processor.h"

class DummyProcessor: public xgboost::processing::Processor {
 private:
    bool active_ = false;
    const std::map<std::string, std::string> *params_;
    std::vector<double> *gh_pairs_{nullptr};
    const xgboost::GHistIndexMatrix *gidx_;

 public:
    void Initialize(bool active, std::map<std::string, std::string> params) override {
        this->active_ = active;
        this->params_ = &params;
    }

    void Shutdown() override {
        this->gh_pairs_ = nullptr;
        this->gidx_ = nullptr;
    }

    void FreeBuffer(xgboost::common::Span<std::int8_t> buffer) override {
        free(buffer.data());
    }

    xgboost::common::Span<int8_t> ProcessGHPairs(std::vector<double> &pairs) override;

    xgboost::common::Span<int8_t> HandleGHPairs(xgboost::common::Span<int8_t> buffer) override;

    void InitAggregationContext(xgboost::GHistIndexMatrix const &gidx) override {
        this->gidx_ = &gidx;
    }

    xgboost::common::Span<std::int8_t> ProcessAggregation(std::vector<xgboost::bst_node_t> const &nodes_to_build,
                                                          xgboost::common::RowSetCollection row_set) override;

    std::vector<double> HandleAggregation(xgboost::common::Span<std::int8_t> buffer) override;
};
