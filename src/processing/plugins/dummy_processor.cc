/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#include <iostream>
#include <cstring>
#include "./dummy_processor.h"

using std::vector;
using std::cout;
using std::endl;

const char kSignature[] = "NVDADAM1";  // DAM (Direct Accessible Marshalling) V1
const int64_t kPrefixLen = 24;

bool ValidDam(void *buffer) {
    return memcmp(buffer, kSignature, strlen(kSignature)) == 0;
}

void* DummyProcessor::ProcessGHPairs(size_t &size, std::vector<double>& pairs) {
    cout << "ProcessGHPairs called with pairs size: " << pairs.size() << endl;

    size = kPrefixLen + pairs.size()*10*8;  // Assume encrypted size is 10x

    int64_t buf_size = size;
    // This memory needs to be freed
    char *buf = static_cast<char *>(calloc(size, 1));
    memcpy(buf, kSignature, strlen(kSignature));
    memcpy(buf + 8, &buf_size, 8);
    memcpy(buf + 16, &processing::kDataTypeGHPairs, 8);

    // Simulate encryption by duplicating value 10 times
    int index = kPrefixLen;
    for (auto value : pairs) {
        for (int i = 0; i < 10; i++) {
            memcpy(buf+index, &value, 8);
            index += 8;
        }
    }

    // Save pairs for future operations
    this->gh_pairs_ = new vector<double>(pairs);

    return buf;
}


void* DummyProcessor::HandleGHPairs(size_t &size, void *buffer, size_t buf_size) {
    cout << "HandleGHPairs called with buffer size: " << buf_size << " Active: " << active_ << endl;

    if (!ValidDam(buffer)) {
        cout << "Invalid buffer received" << endl;
        return buffer;
    }

    // For dummy, this call is used to set gh_pairs for passive sites
    if (!active_) {
        int8_t *ptr = static_cast<int8_t *>(buffer);
        ptr += kPrefixLen;
        double *pairs = reinterpret_cast<double *>(ptr);
        size_t num = (buf_size - kPrefixLen) / 8;
        gh_pairs_ = new vector<double>();
        for (int i = 0; i < num; i += 10) {
            gh_pairs_->push_back(pairs[i]);
        }
        cout << "GH Pairs saved. Size: " << gh_pairs_->size() << endl;
    }

    return buffer;
}

void *DummyProcessor::ProcessAggregation(size_t &size, std::map<int, std::vector<int>> nodes) {
    auto total_bin_size = cuts_.back();
    auto histo_size = total_bin_size*2;
    size = kPrefixLen + 8*histo_size*nodes.size();
    int64_t buf_size = size;
    cout << "ProcessAggregation called with bin size: " << total_bin_size << " Buffer Size: " << buf_size << endl;
    std::int8_t *buf = static_cast<std::int8_t *>(calloc(buf_size, 1));
    memcpy(buf, kSignature, strlen(kSignature));
    memcpy(buf + 8, &buf_size, 8);
    memcpy(buf + 16, &processing::kDataTypeHisto, 8);

    double *histo = reinterpret_cast<double *>(buf + kPrefixLen);
    for ( const auto &node : nodes ) {
        auto rows = node.second;
        for (const auto &row_id : rows) {

            auto num = cuts_.size() - 1;
            for (std::size_t f = 0; f < num; f++) {
                auto slot = slots_[f + num*row_id];
                if (slot < 0) {
                    continue;
                }

                if (slot >= total_bin_size) {
                    cout << "Slot too big, ignored: " << slot << endl;
                    continue;
                }

                if (row_id >= gh_pairs_->size()/2) {
                    cout << "Row ID too big: " << row_id << endl;
                }

                auto g = (*gh_pairs_)[row_id*2];
                auto h = (*gh_pairs_)[row_id*2+1];
                histo[slot*2] += g;
                histo[slot*2+1] += h;
            }
        }
        histo += histo_size;
    }

    return buf;
}

std::vector<double> DummyProcessor::HandleAggregation(void *buffer, size_t buf_size) {
    cout << "HandleAggregation called with buffer size: " << buf_size << endl;
    std::vector<double> result = std::vector<double>();

    int8_t* ptr = static_cast<int8_t *>(buffer);
    auto rest_size = buf_size;

    while (rest_size > kPrefixLen) {
        if (!ValidDam(ptr)) {
            cout << "Invalid buffer at offset " << buf_size - rest_size << endl;
            continue;
        }
        std::int64_t *size_ptr = reinterpret_cast<std::int64_t *>(ptr + 8);
        double *array_start = reinterpret_cast<double *>(ptr + kPrefixLen);
        auto array_size = (*size_ptr - kPrefixLen)/8;
        cout << "Histo size for buffer: " << array_size << endl;
        result.insert(result.end(), array_start, array_start + array_size);
        cout << "Result size: " << result.size() << endl;
        rest_size -= *size_ptr;
        ptr = ptr + *size_ptr;
    }

    cout << "Total histo size: " << result.size() << endl;
    
    return result;
}
