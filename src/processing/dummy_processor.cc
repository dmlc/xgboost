#include "dummy_processor.h"

using namespace std;

void DummyProcessor::Initialize(bool active, std::map<std::string, std::string> params) {
    cout << "Initialize called, params:" << endl;
}

void DummyProcessor::Shutdown() {
    cout << "Shutdown called" << endl;
}

xgboost::common::Span<int8_t> DummyProcessor::ProcessGHPairs(vector<double> &pairs) {
    cout << "ProcessGHPairs called with sample_ids size: " << pairs.size() << endl;
    return xgboost::common::Span<int8_t>();
}

xgboost::common::Span<int8_t> DummyProcessor::HandleEncodedPairs(xgboost::common::Span<int8_t> buffer) {
    cout << "HandleEncodedPairs called with sample_ids size: " << buffer.size() << endl;
    return xgboost::common::Span<int8_t>();
}

void DummyProcessor::InitAggregationContext(vector<int> &sample_feature_bin_ids, vector<int> &feature_bin_sizes,
                                              vector<int> &available_features) {
    cout << "InitAggregationContext called" << endl;

}

xgboost::common::Span<int8_t>
DummyProcessor::ProcessAggregationContext(vector<int> &sample_ids, vector<int> &node_sizes) {
    cout << "ProcessAggregationContext called" << endl;
    return xgboost::common::Span<int8_t>();
}

vector<double> DummyProcessor::HandleAggregation(vector<xgboost::common::Span < std::int8_t>> buffers){
    cout << "HandleAggregation called, params:" << endl;
    return std::vector<double>();
}

