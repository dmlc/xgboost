/*!
 * Copyright 2019 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_JSON_WRITER_EXPERIMENTAL_H_
#define XGBOOST_COMMON_JSON_WRITER_EXPERIMENTAL_H_

#include <algorithm>
#include <cinttypes>
#include <string>
#include <vector>

#include "json_experimental.h"
#include "charconv.h"

#include "xgboost/logging.h"

namespace xgboost {
namespace experimental {

class JsonWriter {
  std::vector<std::string::value_type> buffer_;
  char f2s_buffer_[NumericLimits<float>::kDigit10];
  char i2s_buffer_[NumericLimits<int64_t>::kDigit10];

 public:
  void HandleNull() {
    auto s = buffer_.size();
    buffer_.resize(s + 4);
    buffer_[s + 0] = 'n';
    buffer_[s + 1] = 'u';
    buffer_[s + 2] = 'l';
    buffer_[s + 3] = 'l';
  }
  void HandleTrue() {
    auto s = buffer_.size();
    buffer_.resize(s + 4);
    buffer_[s + 0] = 't';
    buffer_[s + 1] = 'r';
    buffer_[s + 2] = 'u';
    buffer_[s + 3] = 'e';
  }
  void HandleFalse() {
    auto s = buffer_.size();
    buffer_.resize(s + 5);
    buffer_[s + 0] = 'f';
    buffer_[s + 1] = 'a';
    buffer_[s + 2] = 'l';
    buffer_[s + 3] = 's';
    buffer_[s + 4] = 'e';
  }

  void BeginObject() { buffer_.emplace_back('{'); }
  void EndObject() { buffer_.emplace_back('}'); }
  void KeyValue() { buffer_.emplace_back(':'); }
  void Comma() { buffer_.emplace_back(','); }

  void BeginArray() { buffer_.emplace_back('['); }
  void EndArray() { buffer_.emplace_back(']'); }


  void HandleString(ConstStringRef string) {
    std::string buffer;
    buffer.reserve(string.size());

    buffer += '"';
    for (size_t i = 0; i < string.size(); i++) {
      switch (string[i]) {
      case '\\': {
        buffer += u8"\\\\";
        break;
      }
      case '\"': {
        buffer += u8"\\\"";
        break;
      }
      case '\b': {
        buffer += u8"\\b";
        break;
      }
      case '\f': {
        buffer += "\\f";
        break;
      }
      case '\t': {
        buffer += "\\t";
        break;
      }
      case '\r': {
        buffer += "\\r";
        break;
      }
      case '\n': {
        buffer += "\\n";
        break;
      }
      default: {
        buffer += string[i];
        break;
      }
      }
    }
    buffer += '"';

    auto s = buffer_.size();
    buffer_.resize(s + buffer.size());
    std::memcpy(buffer_.data() + s, buffer.data(), buffer.size());
  }
  void HandleFloat(float f) {
    auto ret = to_chars(f2s_buffer_, f2s_buffer_ + NumericLimits<float>::kDigit10, f);
    auto end = ret.ptr;
    CHECK(ret.ec == std::errc());
    auto out_size = end - f2s_buffer_;
    auto ori_size = buffer_.size();
    buffer_.resize(buffer_.size() + out_size);
    std::memcpy(buffer_.data() + ori_size, f2s_buffer_, end - f2s_buffer_);
  }
  void HandleInteger(int64_t i) {
    auto ret = to_chars(i2s_buffer_, i2s_buffer_ + NumericLimits<int64_t>::kDigit10, i);
    auto end = ret.ptr;
    CHECK(ret.ec == std::errc());
    auto digits = std::distance(i2s_buffer_, end);
    auto ori_size = buffer_.size();
    buffer_.resize(ori_size + digits);
    std::memcpy(buffer_.data() + ori_size, i2s_buffer_, digits);
  }

  std::vector<std::string::value_type> const& write(ValueImpl<Document> const& value) {
    return buffer_;
  }

  void TakeResult(std::string *str) {
    str->resize(buffer_.size());
    std::copy(buffer_.cbegin(), buffer_.cend(), str->begin());
  }
};

}  // namespace experimental
}  // namespace xgboost

#endif  // XGBOOST_COMMON_JSON_WRITER_EXPERIMENTAL_H_
