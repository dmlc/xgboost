#include "dummy_processor.h"

using namespace std;

const std::string kSignature = "NVDADAM1"; // DAM (Direct Accessible Marshalling) V1
const int kPrefixLen = 16;

// This dummy class simulate encryption by replicating the original number 10 times

xgboost::common::Span<int8_t> DummyProcessor::ProcessGHPairs(vector<double> &pairs) {
    cout << "ProcessGHPairs called with sample_ids size: " << pairs.size() << endl;
    int buf_size = kPrefixLen + pairs.size()*10*8; // Assume encrypted size is 10x

    // This memory needs to be freed
    char *buf = static_cast<char *>(malloc(buf_size));
    memcpy(buf, kSignature.c_str(), kSignature.size());
    memcpy(buf + 8, &buf_size, 8);
    int index = kPrefixLen;
    for (auto value : pairs) {
        // Copy the number 10 times to simulate HE
        for (int i = 0; i < 10; i++) {
            memcpy(buf+index, &value, 8);
            index += 8;
        }
    }

    this->encrypted_gh_pairs = xgboost::common::Span<int8_t>(reinterpret_cast<int8_t *>(buf), index);

    return this->encrypted_gh_pairs;
}

xgboost::common::Span<int8_t> DummyProcessor::HandleEncodedPairs(xgboost::common::Span<int8_t> buffer) {
    cout << "HandleEncodedPairs called with buffer size: " << buffer.size() << endl;
    // For dummy, this call does nothing
    this->encrypted_gh_pairs = buffer;
    return buffer;
}

void DummyProcessor::InitAggregationContext(vector<int> &sample_feature_bin_ids, vector<int> &feature_bin_sizes,
                                              vector<int> &available_features) {
    cout << "InitAggregationContext called" << endl;
    this->sample_feature_bin_ids = &sample_feature_bin_ids;
    this->feature_bin_sizes = &feature_bin_sizes;
    this->available_features = &available_features;
}

xgboost::common::Span<int8_t> DummyProcessor::ProcessAggregationContext(
        vector<int> &sample_ids, vector<int> &node_sizes) {

    cout << "ProcessAggregationContext called sample_ids size: " << sample_ids.size() <<
        " node_sizes size: " << node_sizes.size() << endl;

    int total_bin_size = 0;
    for (auto n : *feature_bin_sizes) {
        total_bin_size += n;
    }

    int buf_size = kPrefixLen +
            8 + // number of node
            total_bin_size*2*8*10; // Times 2 (g & h) * 80 (encrypted_size)

    cout << "Total bin size is " << total_bin_size << endl;

    signed char *buf = static_cast<signed char *>(malloc(buf_size));
    memcpy(buf, kSignature.c_str(), kSignature.size());
    memcpy(buf + 8, &buf_size, 8);
    int index = kPrefixLen;
    int num_hist = node_sizes.size()*2;

    memcpy(buf+index, &num_hist, 8); // Number of histograms
    index += 8;
    signed char *p = encrypted_gh_pairs.data();
    p = p + kPrefixLen;
    double *encrypted_gh = reinterpret_cast<double *>(p);

    p = buf + index;
    double *histograms = reinterpret_cast<double *>(p);
    int entries = total_bin_size*2*10;
    double zero = 0.0;

    // Initialize histogram memory with 0.0
    double *ptr = histograms;
    for (int i = 0; i < entries; i++) {
        memcpy(ptr, &zero, 8);
        ptr += 8;
    }

    int sample_start = 0;
    for (int n = 0; n < node_sizes.size(); n++) {
        double *hist_start = histograms + 2*n*total_bin_size*10;
        auto node_size = node_sizes[n];
        auto sample_end = sample_start + node_size;
        for (int i = sample_start; i < sample_end; i++) {
            int sample_id = sample_ids[i];
            for (int f = 0; f < total_bin_size; f++) {
                int feature_bin = (*sample_feature_bin_ids)[sample_id*total_bin_size+f];
                if (feature_bin == -1) {
                    continue;
                }

                int g_index = sample_id * total_bin_size * 2 + 2*f;

                double g_value = encrypted_gh_pairs[g_index*10];
                double h_value = encrypted_gh_pairs[(g_index+1)*10];
                for (int k = 1; k < 10; k++) {
                    int h_index = 2*f + k;
                    *(hist_start + h_index) += h_value;
                    *(hist_start + h_index + 1 ) += g_value;
                }

            }
        }
        sample_start = sample_end;
    }

    return xgboost::common::Span<int8_t>(buf, buf_size);
}

vector<double> DummyProcessor::HandleAggregation(vector<xgboost::common::Span < std::int8_t>> buffers){
    cout << "HandleAggregation called, params:" << endl;

    // Only active client can decrypt
    if (!active) {
        return std::vector<double>();
    }

    int total_bin_size = 0;
    for (auto n : *feature_bin_sizes) {
        total_bin_size += n;
    }


    std::vector<double> result = std::vector<double>();

    // Not sure what to return here

    return result;
}

