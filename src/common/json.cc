/*!
 * Copyright 2018 by Contributors
 * \file json.cc
 * \brief JSON serialization of nested key-value store.
 */
#include <istream>
#include <stack>
#include <unordered_map>
#include <dmlc/logging.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/reader.h>
#include "./json.h"

namespace xgboost {
namespace serializer {

class JSONInputHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, JSONInputHandler> {
 public:
  explicit inline JSONInputHandler(NestedKVStore* kvstore);
  ~JSONInputHandler() = default;
  inline bool Null();
  inline bool Bool(bool b);
  inline bool Int(int i);
  inline bool Uint(unsigned u);
  inline bool Int64(int64_t i);
  inline bool Uint64(uint64_t u);
  inline bool Double(double d);
  inline bool String(const char* str, rapidjson::SizeType length, bool copy);
  inline bool StartObject();
  inline bool Key(const char* str, rapidjson::SizeType length, bool copy);
  inline bool EndObject(rapidjson::SizeType memberCount);
  inline bool StartArray();
  inline bool EndArray(rapidjson::SizeType elementCount);
 private:
  NestedKVStore* kvstore_;

  /*** State Diagram for JSON parsing ***/
  enum class State : uint8_t {
    kInit, kObject, kExpectValue, kExpectArrayItem
  };
  State parser_state_;
  std::stack<NestedKVStore> object_context_;
  std::string current_key_;
  std::vector<NestedKVStore> current_array_;
  uint32_t parser_depth_;  // Depth of object currently being parsed (initially 0)
};

NestedKVStore LoadKVStoreFromJSON(std::istream* stream) {
  NestedKVStore result;
  JSONInputHandler handler(&result);
  rapidjson::Reader reader;
  rapidjson::IStreamWrapper isw(*stream);
  reader.Parse(isw, handler);
  return result;
}

void SaveKVStoreToJSON(const NestedKVStore& kvstore, std::ostream* stream) {
}

inline JSONInputHandler::JSONInputHandler(NestedKVStore* kvstore)
  : kvstore_(kvstore), parser_state_(JSONInputHandler::State::kInit),
    parser_depth_(0U) {}

inline bool
JSONInputHandler::Null() {
  State next_state = State::kInit;
  // perform transition
  switch (parser_state_) {
   case State::kExpectValue:
    CHECK(!current_key_.empty());
    object_context_.top()[current_key_] = serializer::Null();
    next_state = State::kObject;
    break;
   case State::kExpectArrayItem:
    current_array_.push_back(serializer::Null());
    next_state = State::kExpectArrayItem;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  parser_state_ = next_state;
  return true;
}

inline bool
JSONInputHandler::Bool(bool b) {
  State next_state = State::kInit;
  // perform transition
  switch (parser_state_) {
   case State::kExpectValue:
    CHECK(!current_key_.empty());
    object_context_.top()[current_key_] = b;
    next_state = State::kObject;
    break;
   case State::kExpectArrayItem:
    current_array_.push_back(serializer::Boolean(b));
    next_state = State::kExpectArrayItem;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  parser_state_ = next_state;
  return true;
}

inline bool
JSONInputHandler::Int64(int64_t i) {
  State next_state = State::kInit;
  // perform transition
  switch (parser_state_) {
   case State::kExpectValue:
    CHECK(!current_key_.empty());
    object_context_.top()[current_key_] = i;
    next_state = State::kObject;
    break;
   case State::kExpectArrayItem:
    current_array_.push_back(serializer::Integer(i));
    next_state = State::kExpectArrayItem;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  parser_state_ = next_state;
  return true;
}

inline bool
JSONInputHandler::Uint64(uint64_t u) {
  return Int64(static_cast<int64_t>(u));
}

inline bool
JSONInputHandler::Int(int i) {
  return Int64(static_cast<int64_t>(i));
}

inline bool
JSONInputHandler::Uint(unsigned u) {
  return Int64(static_cast<int64_t>(u));
}

inline bool
JSONInputHandler::Double(double d) {
  State next_state = State::kInit;
  // perform transition
  switch (parser_state_) {
   case State::kExpectValue:
    CHECK(!current_key_.empty());
    object_context_.top()[current_key_] = d;
    next_state = State::kObject;
    break;
   case State::kExpectArrayItem:
    current_array_.push_back(serializer::Number(d));
    next_state = State::kExpectArrayItem;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  parser_state_ = next_state;
  return true;
}

inline bool
JSONInputHandler::String(const char* str, rapidjson::SizeType length, bool copy) {
  const std::string value(str, static_cast<size_t>(length));
  State next_state = State::kInit;
  // perform transition
  switch (parser_state_) {
   case State::kExpectValue:
    CHECK(!current_key_.empty());
    object_context_.top()[current_key_] = value;
    next_state = State::kObject;
    break;
   case State::kExpectArrayItem:
    current_array_.push_back(serializer::String(value));
    next_state = State::kExpectArrayItem;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  parser_state_ = next_state;
  return true;
}

inline bool
JSONInputHandler::StartObject() {
  State next_state = State::kInit;
  uint32_t next_parser_depth = parser_depth_;
  // perform transition
  switch (parser_state_) {
   case State::kInit:
    next_state = State::kObject;
    next_parser_depth = 0;
    (*kvstore_) = serializer::Object();
    object_context_.push(*kvstore_);
    break;
   case State::kExpectValue:
    CHECK(!current_key_.empty());
    next_state = State::kObject;
    ++next_parser_depth;
    {
      auto obj = serializer::Object();
      object_context_.top()[current_key_] = obj;
      object_context_.push(obj);
    }
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  current_key_ = std::string();
  parser_state_ = next_state;
  parser_depth_ = next_parser_depth;
  return true;
}

inline bool
JSONInputHandler::Key(const char* str, rapidjson::SizeType length, bool copy) {
  State next_state = State::kInit;
  // perform transition
  switch (parser_state_) {
   case State::kObject:
    next_state = State::kExpectValue;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  current_key_ = std::string(str, length);
  parser_state_ = next_state;
  return true;
}

inline bool
JSONInputHandler::EndObject(rapidjson::SizeType memberCount) {
  CHECK_GE(parser_depth_, 1);
  State next_state = State::kInit;
  // perform transition
  switch (parser_state_) {
   case State::kObject:
    next_state = State::kObject;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  object_context_.pop();
  current_key_ = std::string();
  parser_state_ = next_state;
  --parser_depth_;
  return true;
}

inline bool
JSONInputHandler::StartArray() {
  State next_state = State::kInit;
  // perform transition
  switch (parser_state_) {
   case State::kExpectValue:
    CHECK(current_array_.empty());
    next_state = State::kExpectArrayItem;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  current_array_.clear();
  parser_state_ = next_state;
  ++parser_depth_;
  return true;
}

inline bool
JSONInputHandler::EndArray(rapidjson::SizeType elementCount) {
  CHECK_GE(parser_depth_, 1);
  State next_state = State::kInit;
  // perform transition
  switch (parser_state_) {
   case State::kExpectArrayItem:
    next_state = State::kObject;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  CHECK(!current_key_.empty());
  object_context_.top()[current_key_] = current_array_;
  object_context_.pop();
  parser_state_ = next_state;
  --parser_depth_;
  return true;
}

}   // namespace serializer
}   // namespace xgboost
