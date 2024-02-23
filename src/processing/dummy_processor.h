#pragma once

#include "processor.h"

class DummyProcessor: public xgboost::processing::Processor {

    void Initialize(bool active, std::map<std::string, std::string> params) override;

    void Shutdown() override;

    xgboost::common::Span<int8_t> ProcessGHPairs(std::vector<double> &pairs) override;

    xgboost::common::Span<int8_t> HandleEncodedPairs(xgboost::common::Span<int8_t> buffer) override;

    void InitAggregationContext(std::vector<int> &sample_feature_bin_ids, std::vector<int> &feature_bin_sizes,
                                std::vector<int> &available_features) override;

    xgboost::common::Span<int8_t> ProcessAggregationContext(
            std::vector<int> &sample_ids, std::vector<int> &node_sizes) override;

    std::vector<double> HandleAggregation(std::vector<xgboost::common::Span < std::int8_t>> buffers) override;

};