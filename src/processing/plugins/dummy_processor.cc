/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#include "./dummy_processor.h"

using std::vector;
using std::cout;
using std::endl;

const char kSignature[] = "NVDADAM1";  // DAM (Direct Accessible Marshalling) V1
const int kPrefixLen = 24;

xgboost::common::Span<int8_t> DummyProcessor::ProcessGHPairs(vector<double> &pairs) {
    cout << "ProcessGHPairs called with pairs size: " << pairs.size() << endl;

    auto buf_size = kPrefixLen + pairs.size()*10*8;  // Assume encrypted size is 10x

    // This memory needs to be freed
    char *buf = static_cast<char *>(calloc(buf_size, 1));
    memcpy(buf, kSignature, strlen(kSignature));
    memcpy(buf + 8, &buf_size, 8);
    memcpy(buf + 16, &xgboost::processing::kDataTypeGHPairs, 8);

    // Simulate encryption by duplicating value 10 times
    int index = kPrefixLen;
    for (auto value : pairs) {
        for (int i = 0; i < 10; i++) {
            memcpy(buf+index, &value, 8);
            index += 8;
        }
    }

    // Save pairs for future operations
    this->gh_pairs_ = &pairs;

    return xgboost::common::Span<int8_t>(reinterpret_cast<int8_t *>(buf), buf_size);
}

xgboost::common::Span<int8_t> DummyProcessor::HandleGHPairs(xgboost::common::Span<int8_t> buffer) {
    cout << "HandleGHPairs called with buffer size: " << buffer.size() << endl;

    // For dummy, this call is used to set gh_pairs for passive sites
    if (!active_) {
        int8_t *ptr = buffer.data() + kPrefixLen;
        double *pairs = reinterpret_cast<double *>(ptr);
        size_t num = (buffer.size() - kPrefixLen) / 8;
        gh_pairs_ = new vector<double>(pairs, pairs + num);
    }

    return buffer;
}

xgboost::common::Span<std::int8_t> DummyProcessor::ProcessAggregation(
        std::vector<xgboost::bst_node_t> const &nodes_to_build, xgboost::common::RowSetCollection const &row_set) {
    auto total_bin_size = gidx_->Cuts().Values().size();
    auto histo_size = total_bin_size*2;
    auto buf_size = kPrefixLen + 8*histo_size*nodes_to_build.size();
    std::int8_t *buf = static_cast<std::int8_t *>(calloc(buf_size, 1));
    memcpy(buf, kSignature, strlen(kSignature));
    memcpy(buf + 8, &buf_size, 8);
    memcpy(buf + 16, &xgboost::processing::kDataTypeHisto, 8);

    double *histo = reinterpret_cast<double *>(buf + kPrefixLen);
    for (auto &node_id : nodes_to_build) {
        auto elem = row_set[node_id];
        for (auto it = elem.begin; it != elem.end; ++it) {
            auto row_id = *it;
            for (std::size_t f = 0; f < gidx_->Cuts().Ptrs().size()-1; f++) {
                auto slot = gidx_->GetGindex(row_id, f);
                if (slot < 0) {
                    continue;
                }

                auto g = (*gh_pairs_)[row_id*2];
                auto h = (*gh_pairs_)[row_id*2+1];
                histo[slot*2] += g;
                histo[slot*2+1] += h;
            }
        }
        histo += histo_size;
    }

    return xgboost::common::Span<int8_t>(reinterpret_cast<int8_t *>(buf), buf_size);
}

std::vector<double> DummyProcessor::HandleAggregation(xgboost::common::Span<std::int8_t> buffer) {
    std::vector<double> result = std::vector<double>();

    int8_t* ptr = buffer.data();
    auto rest_size = buffer.size();

    while (rest_size > kPrefixLen) {
        std::int64_t *size_ptr = reinterpret_cast<std::int64_t *>(ptr + 8);
        double *array_start = reinterpret_cast<double *>(ptr + kPrefixLen);
        auto array_size = (*size_ptr - kPrefixLen)/8;
        result.insert(result.end(), array_start, array_start + array_size);

        rest_size -= *size_ptr;
        ptr = ptr + *size_ptr;
    }

    return result;
}

