/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#include <iostream>
#include <cstring>
#include <cstdint>
#include "./mock_processor.h"

const char kSignature[] = "NVDADAM1";  // DAM (Direct Accessible Marshalling) V1
const int64_t kPrefixLen = 24;

bool ValidDam(void *buffer, std::size_t size) {
  return size >= kPrefixLen && memcmp(buffer, kSignature, strlen(kSignature)) == 0;
}

void* MockProcessor::ProcessGHPairs(std::size_t *size, const std::vector<double>& pairs) {
  *size = kPrefixLen + pairs.size()*10*8;  // Assume encrypted size is 10x

  int64_t buf_size = *size;
  // This memory needs to be freed
  char *buf = static_cast<char *>(calloc(*size, 1));
  memcpy(buf, kSignature, strlen(kSignature));
  memcpy(buf + 8, &buf_size, 8);
  memcpy(buf + 16, &kDataTypeGHPairs, 8);

  // Simulate encryption by duplicating value 10 times
  int index = kPrefixLen;
  for (auto value : pairs) {
    for (std::size_t i = 0; i < 10; i++) {
      memcpy(buf+index, &value, 8);
      index += 8;
    }
  }

  // Save pairs for future operations
  this->gh_pairs_ = new std::vector<double>(pairs);

  return buf;
}


void* MockProcessor::HandleGHPairs(std::size_t *size, void *buffer, std::size_t buf_size) {
  *size = buf_size;
  if (!ValidDam(buffer, *size)) {
    return buffer;
  }

  // For mock, this call is used to set gh_pairs for passive sites
  if (!active_) {
    int8_t *ptr = static_cast<int8_t *>(buffer);
    ptr += kPrefixLen;
    double *pairs = reinterpret_cast<double *>(ptr);
    std::size_t num = (buf_size - kPrefixLen) / 8;
    gh_pairs_ = new std::vector<double>();
    for (std::size_t i = 0; i < num; i += 10) {
      gh_pairs_->push_back(pairs[i]);
    }
  }

  auto result = malloc(buf_size);
  memcpy(result, buffer, buf_size);

  return result;
}

void *MockProcessor::ProcessAggregation(std::size_t *size, std::map<int, std::vector<int>> nodes) {
  int total_bin_size = cuts_.back();
  int histo_size = total_bin_size*2;
  *size = kPrefixLen + 8*histo_size*nodes.size();
  int64_t buf_size = *size;
  int8_t *buf = static_cast<int8_t *>(calloc(buf_size, 1));
  memcpy(buf, kSignature, strlen(kSignature));
  memcpy(buf + 8, &buf_size, 8);
  memcpy(buf + 16, &kDataTypeHisto, 8);

  double *histo = reinterpret_cast<double *>(buf + kPrefixLen);
  for ( const auto &node : nodes ) {
    auto rows = node.second;
    for (const auto &row_id : rows) {
      auto num = cuts_.size() - 1;
      for (std::size_t f = 0; f < num; f++) {
        int slot = slots_[f + num*row_id];
        if ((slot < 0) || (slot >= total_bin_size)) {
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

  return buf;
}

std::vector<double> MockProcessor::HandleAggregation(void *buffer, std::size_t buf_size) {
  std::vector<double> result = std::vector<double>();

  int8_t* ptr = static_cast<int8_t *>(buffer);
  auto rest_size = buf_size;

  while (rest_size > kPrefixLen) {
    if (!ValidDam(ptr, rest_size)) {
        break;
    }
    int64_t *size_ptr = reinterpret_cast<int64_t *>(ptr + 8);
    double *array_start = reinterpret_cast<double *>(ptr + kPrefixLen);
    auto array_size = (*size_ptr - kPrefixLen)/8;
    result.insert(result.end(), array_start, array_start + array_size);
    rest_size -= *size_ptr;
    ptr = ptr + *size_ptr;
  }

  return result;
}
