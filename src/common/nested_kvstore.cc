/*!
 * Copyright 2018 by Contributors
 * \file nested_kvstore.cc
 * \brief Simple implementation of nested key-value store.
 */
#include <dmlc/logging.h>
#include <xgboost/base.h>

#include <locale>
#include <cctype>  // isdigit, isspace
#include "nested_kvstore.h"

namespace xgboost {
namespace serializer {

// Value
std::string Value::TypeStr() const {
  switch (kind_) {
    case ValueKind::kString:  return "String";  break;
    case ValueKind::kInteger: return "Integer"; break;
    case ValueKind::kNumber:  return "Number";  break;
    case ValueKind::kObject:  return "Object";  break;
    case ValueKind::kArray:   return "Array";   break;
    case ValueKind::kBoolean: return "Boolean"; break;
    case ValueKind::kNull:    return "Null";    break;
  }
  return "";
}

// object value
ObjectValue::ObjectValue() : Value(ValueKind::kObject) {}
ObjectValue::ObjectValue(std::map<std::string, NestedKVStore>&& object)
    : Value(ValueKind::kObject), object_(std::move(object)) {}
ObjectValue::ObjectValue(std::map<std::string, NestedKVStore> const& object)
    : Value(ValueKind::kObject), object_(object) {}

NestedKVStore& ObjectValue::operator[](std::string const& key) {
  return object_[key];
}

// Only used for keeping old compilers happy about non-reaching return
// statement.
NestedKVStore& DummyObjectValue() {
  static NestedKVStore obj;
  return obj;
}

// object
NestedKVStore& ObjectValue::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr()
             << " can not be indexed by integer.";
  return DummyObjectValue();
}

NestedKVStore& ObjectValue::append(NestedKVStore val) {
  LOG(FATAL) << "Cannot append a value to a non-list Object, "
             << "which is actually of type " << Value::TypeStr();
  return DummyObjectValue();
}

bool ObjectValue::operator==(Value const& rhs) const {
  if ( !IsA<ObjectValue>(&rhs) ) { return false; }
  if (object_.size() != Cast<ObjectValue const>(&rhs)->GetObject().size()) {
    return false;
  }
  bool result = object_ == Cast<ObjectValue const>(&rhs)->GetObject();
  return result;
}

// string value
NestedKVStore& StringValue::operator[](std::string const & key) {
  throw std::runtime_error(
      "Object of type " +
      Value::TypeStr() + " can not be indexed by string.");
  return DummyObjectValue();
}

NestedKVStore& StringValue::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr()
             << " can not be indexed by integer."
             << "please try obtaining std::string first.";
  return DummyObjectValue();
}

NestedKVStore& StringValue::append(NestedKVStore val) {
  LOG(FATAL) << "Cannot append a value to a non-list Object, "
             << "which is actually of type " << Value::TypeStr();
  return DummyObjectValue();
}

bool StringValue::operator==(Value const& rhs) const {
  if (!IsA<StringValue>(&rhs)) { return false; }
  return Cast<StringValue const>(&rhs)->GetString() == str_;
}

// array value
ArrayValue::ArrayValue() : Value(ValueKind::kArray) {}
ArrayValue::ArrayValue(std::vector<NestedKVStore>&& arr) :
    Value(ValueKind::kArray), vec_(std::move(arr)) {}
ArrayValue::ArrayValue(std::vector<NestedKVStore> const& arr) :
    Value(ValueKind::kArray), vec_(arr) {}

NestedKVStore& ArrayValue::operator[](std::string const & key) {
  throw std::runtime_error(
      "Object of type " +
      Value::TypeStr() + " can not be indexed by string.");
  return DummyObjectValue();
}

NestedKVStore& ArrayValue::operator[](int ind) {
  return vec_.at(ind);
}

NestedKVStore& ArrayValue::append(NestedKVStore val) {
  vec_.push_back(val);
  return vec_.back();
}

bool ArrayValue::operator==(Value const& rhs) const {
  if (!IsA<ArrayValue>(&rhs)) { return false; }
  auto& arr = Cast<ArrayValue const>(&rhs)->GetArray();
  if (arr.size() != vec_.size()) { return false; }
  return std::equal(arr.cbegin(), arr.cend(), vec_.cbegin());
}

// integer value
NestedKVStore& IntegerValue::operator[](std::string const & key) {
  throw std::runtime_error(
      "Object of type " +
      Value::TypeStr() + " can not be indexed by string.");
  return DummyObjectValue();
}

NestedKVStore& IntegerValue::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr()
             << " can not be indexed by integer.";
  return DummyObjectValue();
}

NestedKVStore& IntegerValue::append(NestedKVStore val) {
  LOG(FATAL) << "Cannot append a value to a non-list Object, "
             << "which is actually of type " << Value::TypeStr();
  return DummyObjectValue();
}

bool IntegerValue::operator==(Value const& rhs) const {
  if (!IsA<IntegerValue>(&rhs)) { return false; }
  double residue =
      std::abs(number_ - Cast<IntegerValue const>(&rhs)->GetInteger());
  return residue < kRtEps;
}

int64_t IntegerValue::GetInteger() const {
  return number_;
}

// number value
NestedKVStore& NumberValue::operator[](std::string const & key) {
  throw std::runtime_error(
      "Object of type " +
      Value::TypeStr() + " can not be indexed by string.");
  return DummyObjectValue();
}

NestedKVStore& NumberValue::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr()
             << " can not be indexed by integer.";
  return DummyObjectValue();
}

NestedKVStore& NumberValue::append(NestedKVStore val) {
  LOG(FATAL) << "Cannot append a value to a non-list Object, "
             << "which is actually of type " << Value::TypeStr();
  return DummyObjectValue();
}

bool NumberValue::operator==(Value const& rhs) const {
  if (!IsA<NumberValue>(&rhs)) { return false; }
  double residue =
      std::abs(number_ - Cast<NumberValue const>(&rhs)->GetDouble());
  return residue < kRtEps;
}

float NumberValue::GetFloat() const {
  return static_cast<float>(number_);
}

double NumberValue::GetDouble() const {
  return number_;
}

// null value
NestedKVStore& NullValue::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr()
             << " can not be indexed by string.";
  return DummyObjectValue();
}

NestedKVStore& NullValue::operator[](int ind) {
  throw std::runtime_error(
      "Object of type " +
      Value::TypeStr() + " can not be indexed by integer.");
  return DummyObjectValue();
}

NestedKVStore& NullValue::append(NestedKVStore val) {
  LOG(FATAL) << "Cannot append a value to a non-list Object, "
             << "which is actually of type " << Value::TypeStr();
  return DummyObjectValue();
}

bool NullValue::operator==(Value const& rhs) const {
  if ( !IsA<NullValue>(&rhs) ) { return false; }
  return true;
}

// boolean value
NestedKVStore& BooleanValue::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr()
             << " can not be indexed by string.";
  return DummyObjectValue();
}

NestedKVStore& BooleanValue::operator[](int ind) {
  throw std::runtime_error(
      "Object of type " +
      Value::TypeStr() + " can not be indexed by Integer.");
  return DummyObjectValue();
}

NestedKVStore& BooleanValue::append(NestedKVStore val) {
  LOG(FATAL) << "Cannot append a value to a non-list Object, "
             << "which is actually of type " << Value::TypeStr();
  return DummyObjectValue();
}

bool BooleanValue::operator==(Value const& rhs) const {
  if (!IsA<BooleanValue>(&rhs)) { return false; }
  return boolean_ == Cast<BooleanValue const>(&rhs)->GetBoolean();
}

}  // namespace serializer
}  // namespace xgboost
