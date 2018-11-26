/*!
 * Copyright 2018 by Contributors
 * \file json.cc
 * \brief JSON serialization of nested key-value store. Uses Tencent/RapidJSON
 */
#include <istream>
#include <stack>
#include <limits>
#include <unordered_map>
#include <dmlc/logging.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/reader.h>
#include <rapidjson/writer.h>
#include "./json.h"

namespace xgboost {
namespace serializer {

/* Use RapidJSON SAX API to parser JSON into nested key-value store. See
 * http://rapidjson.org/md_doc_sax.html for more information. */
class JSONInputHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, JSONInputHandler> {
 public:
  explicit JSONInputHandler(NestedKVStore* kvstore);
  ~JSONInputHandler() = default;
  bool Null();
  bool Bool(bool b);
  bool Int(int i);
  bool Uint(unsigned u);
  bool Int64(int64_t i);
  bool Uint64(uint64_t u);
  bool Double(double d);
  bool String(const char* str, rapidjson::SizeType length, bool copy);
  bool StartObject();
  bool Key(const char* str, rapidjson::SizeType length, bool copy);
  bool EndObject(rapidjson::SizeType memberCount);
  bool StartArray();
  bool EndArray(rapidjson::SizeType elementCount);

  void CheckPostCondition();
 private:
  NestedKVStore* kvstore_;

  enum class State : uint8_t {
    kInit, kObject, kExpectValue, kExpectArrayItem
  };
  State parser_state_;
  std::stack<NestedKVStore> object_context_;
  std::string current_key_;
  uint32_t parser_depth_;  // Depth of object currently being parsed (initially 0)
};

NestedKVStore LoadKVStoreFromJSON(std::istream* stream) {
  NestedKVStore result;
  JSONInputHandler handler(&result);
  rapidjson::Reader reader;
  rapidjson::IStreamWrapper isw(*stream);
  CHECK(reader.Parse(isw, handler));
  handler.CheckPostCondition();
  return result;
}

template <typename OutputStreamType>
void SaveKVStoreToJSON_(const NestedKVStore& kvstore,
                        rapidjson::Writer<OutputStreamType>* writer);

void SaveKVStoreToJSON(const NestedKVStore& kvstore, std::ostream* stream) {
  rapidjson::OStreamWrapper osw(*stream);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
  writer.SetMaxDecimalPlaces(std::numeric_limits<double>::max_digits10);
  SaveKVStoreToJSON_(kvstore, &writer);
}

template <typename OutputStreamType>
void SaveKVStoreToJSON_(const NestedKVStore& kvstore,
                        rapidjson::Writer<OutputStreamType>* writer) {
  switch (kvstore.GetValue().Type()) {
   case Value::ValueKind::kString: {
      const std::string& value = Get<String>(kvstore).GetString();
      writer->String(value.c_str(), value.length());
    }
    break;
   case Value::ValueKind::kNumber:
    writer->Double(Get<Number>(kvstore).GetDouble());
    break;
   case Value::ValueKind::kInteger:
    writer->Int64(Get<Integer>(kvstore).GetInteger());
    break;
   case Value::ValueKind::kObject: {
      const auto& map = Get<Object>(kvstore).GetObject();
      writer->StartObject();
      for (const auto& kv : map) {
        writer->Key(kv.first.c_str(), kv.first.length());
        SaveKVStoreToJSON_(kv.second, writer);
      }
      writer->EndObject();
    }
    break;
   case Value::ValueKind::kArray: {
      const auto& array = Get<Array>(kvstore).GetArray();
      writer->StartArray();
      for (const auto& e : array) {
        SaveKVStoreToJSON_(e, writer);
      }
      writer->EndArray();
    }
    break;
   case Value::ValueKind::kBoolean:
    writer->Bool(Get<Boolean>(kvstore).GetBoolean());
    break;
   case Value::ValueKind::kNull:
    writer->Null();
    break;
  }
}

JSONInputHandler::JSONInputHandler(NestedKVStore* kvstore)
  : kvstore_(kvstore), parser_state_(JSONInputHandler::State::kInit),
    parser_depth_(1U) {}

void
JSONInputHandler::CheckPostCondition() {
  CHECK_EQ(parser_depth_, 0);
  CHECK(parser_state_ == State::kObject);
}

bool
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
    object_context_.top().append(serializer::Null());
    next_state = State::kExpectArrayItem;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  parser_state_ = next_state;
  return true;
}

bool
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
    object_context_.top().append(serializer::Boolean(b));
    next_state = State::kExpectArrayItem;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  parser_state_ = next_state;
  return true;
}

bool
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
    object_context_.top().append(serializer::Integer(i));
    next_state = State::kExpectArrayItem;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  parser_state_ = next_state;
  return true;
}

bool
JSONInputHandler::Uint64(uint64_t u) {
  return Int64(static_cast<int64_t>(u));
}

bool
JSONInputHandler::Int(int i) {
  return Int64(static_cast<int64_t>(i));
}

bool
JSONInputHandler::Uint(unsigned u) {
  return Int64(static_cast<int64_t>(u));
}

bool
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
    object_context_.top().append(serializer::Number(d));
    next_state = State::kExpectArrayItem;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  parser_state_ = next_state;
  return true;
}

bool
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
    object_context_.top().append(serializer::String(value));
    next_state = State::kExpectArrayItem;
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  parser_state_ = next_state;
  return true;
}

bool
JSONInputHandler::StartObject() {
  State next_state = State::kInit;
  uint32_t next_parser_depth = parser_depth_;
  // perform transition
  switch (parser_state_) {
   case State::kInit:
    next_state = State::kObject;
    next_parser_depth = 1;
    (*kvstore_) = serializer::Object();
    object_context_.push(*kvstore_);
    break;
   case State::kExpectValue:
    CHECK(!current_key_.empty());
    next_state = State::kObject;
    ++next_parser_depth;
    {
      NestedKVStore obj = serializer::Object();
      object_context_.top()[current_key_] = obj;
      object_context_.push(obj);
    }
    break;
   case State::kExpectArrayItem:
    next_state = State::kObject;
    ++next_parser_depth;
    {
      NestedKVStore obj = serializer::Object();
      object_context_.top().append(obj);
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

bool
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

bool
JSONInputHandler::EndObject(rapidjson::SizeType memberCount) {
  CHECK_GE(parser_depth_, 1);
  State next_state = State::kInit;
  // perform transition
  switch (parser_state_) {
   case State::kObject:
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  object_context_.pop();
  if (object_context_.empty()) {
    CHECK_EQ(parser_depth_, 1);
    next_state = State::kObject;
  } else {
    switch(object_context_.top().GetValue().Type()) {
     case Value::ValueKind::kArray:
      next_state = State::kExpectArrayItem;
      break;
     case Value::ValueKind::kObject:
      next_state = State::kObject;
      break;
     default:
      LOG(FATAL) << "Illegal transition detected";
    }
  }
  current_key_ = std::string();
  parser_state_ = next_state;
  --parser_depth_;
  return true;
}

bool
JSONInputHandler::StartArray() {
  State next_state = State::kInit;
  uint32_t next_parser_depth = parser_depth_;
  // perform transition
  switch (parser_state_) {
   case State::kInit:
    next_parser_depth = 1;
    (*kvstore_) = serializer::Array();
    object_context_.push(*kvstore_);
    break;
   case State::kExpectValue:
    CHECK(!current_key_.empty());
    ++next_parser_depth;
    {
      NestedKVStore obj = serializer::Array();
      object_context_.top()[current_key_] = obj;
      object_context_.push(obj);
    }
    break;
   case State::kExpectArrayItem:
    ++next_parser_depth;
    {
      NestedKVStore obj = serializer::Array();
      object_context_.top().append(obj);
      object_context_.push(obj);
    }
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  current_key_ = std::string();
  parser_state_ = State::kExpectArrayItem;
  parser_depth_ = next_parser_depth;
  return true;
}

bool
JSONInputHandler::EndArray(rapidjson::SizeType elementCount) {
  CHECK_GE(parser_depth_, 1);
  State next_state = State::kInit;
  // perform transition
  switch (parser_state_) {
   case State::kExpectArrayItem:
    break;
   default:
    LOG(FATAL) << "Illegal transition detected";
  }
  // move to next state
  object_context_.pop();
  if (object_context_.empty()) {
    CHECK_EQ(parser_depth_, 1);
    next_state = State::kObject;
  } else {
    switch(object_context_.top().GetValue().Type()) {
     case Value::ValueKind::kArray:
      next_state = State::kExpectArrayItem;
      break;
     case Value::ValueKind::kObject:
      next_state = State::kObject;
      break;
     default:
      LOG(FATAL) << "Illegal transition detected";
    }
  }
  parser_state_ = next_state;
  --parser_depth_;
  return true;
}

}   // namespace serializer
}   // namespace xgboost
