/*!
 * Copyright 2018 by Contributors
 * \file nested_kvstore.h
 * \brief Simple implementation of nested key-value store.
 */
#ifndef XGBOOST_COMMON_NESTED_KVSTORE_H_
#define XGBOOST_COMMON_NESTED_KVSTORE_H_

#include <iostream>
#include <istream>
#include <string>
#include <sstream>
#include <type_traits>

#include <map>
#include <memory>
#include <vector>

namespace xgboost {
namespace serializer {

class NestedKVStore;

class Value {
 public:
  /*!\brief Simplified implementation of LLVM RTTI. */
  enum class ValueKind {
    kString,
    kNumber,
    kInteger,
    kObject,  // std::map
    kArray,   // std::vector
    kBoolean,
    kNull
  };

 private:
  ValueKind kind_;

 public:
  explicit Value(ValueKind _kind) : kind_{_kind} {}

  ValueKind Type() const { return kind_; }
  virtual ~Value() = default;

  virtual NestedKVStore& operator[](std::string const & key) = 0;
  virtual NestedKVStore& operator[](int ind) = 0;
  virtual NestedKVStore& append(NestedKVStore val) = 0;

  virtual bool operator==(Value const& rhs) const = 0;

  std::string TypeStr() const;
};

template <typename T>
bool IsA(Value const* value) {
  return T::IsClassOf(value);
}

template <typename T, typename U>
T* Cast(U* value) {
  if (IsA<T>(value)) {
    return dynamic_cast<T*>(value);
  } else {
    throw std::runtime_error(
        "Invalid cast, from " + value->TypeStr() + " to " + T().TypeStr());
  }
}

class StringValue : public Value {
  std::string str_;
 public:
  StringValue() : Value(ValueKind::kString) {}
  StringValue(std::string const& str) :  // NOLINT
      Value(ValueKind::kString), str_(str) {}
  StringValue(std::string&& str) :  // NOLINT
      Value(ValueKind::kString), str_(std::move(str)) {}

  NestedKVStore& operator[](std::string const & key) override;
  NestedKVStore& operator[](int ind) override;
  NestedKVStore& append(NestedKVStore val) override;

  bool operator==(Value const& rhs) const override;

  std::string const& GetString() const { return str_; }
  std::string & GetString() { return str_;}

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kString;
  }
};

class ArrayValue : public Value {
  std::vector<NestedKVStore> vec_;
 public:
  ArrayValue();
  ArrayValue(std::vector<NestedKVStore>&& arr);        // NOLINT
  ArrayValue(std::vector<NestedKVStore> const& arr);   // NOLINT

  NestedKVStore& operator[](std::string const & key) override;
  NestedKVStore& operator[](int ind) override;
  NestedKVStore& append(NestedKVStore val) override;

  std::vector<NestedKVStore> const& GetArray() const { return vec_; }
  std::vector<NestedKVStore> & GetArray() { return vec_; }

  bool operator==(Value const& rhs) const override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kArray;
  }
};

class ObjectValue : public Value {
  std::map<std::string, NestedKVStore> object_;

 public:
  ObjectValue();   // NOLINT
  ObjectValue(std::map<std::string, NestedKVStore>&& object);  // NOLINT
  ObjectValue(std::map<std::string, NestedKVStore> const& object);  // NOLINT

  NestedKVStore& operator[](std::string const & key) override;
  NestedKVStore& operator[](int ind) override;
  NestedKVStore& append(NestedKVStore val) override;

  bool operator==(Value const& rhs) const override;

  std::map<std::string, NestedKVStore> const& GetObject() const { return object_; }
  std::map<std::string, NestedKVStore> &      GetObject()       { return object_; }

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kObject;
  }
};

class IntegerValue : public Value {
  int64_t number_;

 public:
  IntegerValue() : Value(ValueKind::kInteger), number_{0} {}
  // Ambigious with BooleanValue / NumberValue.
  template <typename IntegerType,
            typename std::enable_if<
              std::is_integral<IntegerType>::value &&
              (!std::is_same<IntegerType, bool>::value) &&
              (!std::is_same<IntegerType, bool const>::value)>::type* = nullptr>
  IntegerValue(IntegerType value) :   // NOLINT
    Value(ValueKind::kInteger), number_(static_cast<int64_t>(value)) {}

  NestedKVStore& operator[](std::string const & key) override;
  NestedKVStore& operator[](int ind) override;
  NestedKVStore& append(NestedKVStore val) override;

  bool operator==(Value const& rhs) const override;

  int64_t GetInteger() const;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kInteger;
  }
};

class NumberValue : public Value {
  double number_;

 public:
  NumberValue() : Value(ValueKind::kNumber), number_{0} {}
  // Ambigious with BooleanValue / IntegerValue.
  template <typename NumberType,
            typename std::enable_if<
              std::is_floating_point<NumberType>::value>::type* = nullptr>
  NumberValue(NumberType value) :    // NOLINT
    Value(ValueKind::kNumber), number_(static_cast<double>(value)) {}

  NestedKVStore& operator[](std::string const & key) override;
  NestedKVStore& operator[](int ind) override;
  NestedKVStore& append(NestedKVStore val) override;

  bool operator==(Value const& rhs) const override;

  float GetFloat() const;
  double GetDouble() const;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kNumber;
  }
};

class NullValue : public Value {
 public:
  NullValue() : Value(ValueKind::kNull) {}

  NestedKVStore& operator[](std::string const & key) override;
  NestedKVStore& operator[](int ind) override;
  NestedKVStore& append(NestedKVStore val) override;

  bool operator==(Value const& rhs) const override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kNull;
  }
};

/*! \brief Describes both true and false. */
class BooleanValue : public Value {
  bool boolean_;
 public:
  BooleanValue() : Value(ValueKind::kBoolean) {}
  // Ambigious with NumberValue / IntegerValue.
  template <typename Bool,
            typename std::enable_if<
              std::is_same<Bool, bool>::value ||
              std::is_same<Bool, bool const>::value>::type* = nullptr>
  BooleanValue(Bool value) :  // NOLINT
      Value(ValueKind::kBoolean), boolean_(value) {}

  NestedKVStore& operator[](std::string const & key) override;
  NestedKVStore& operator[](int ind) override;
  NestedKVStore& append(NestedKVStore val) override;

  bool operator==(Value const& rhs) const override;

  bool GetBoolean() const { return boolean_; }

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kBoolean;
  }
};

/*!
 * \brief Nested key-value store
 *
 * Examples:
 *
 * \code
 *   // Create a nested key-value store
 *   serializer::NestedKVStore object = NestedKVStore::Object();
 *   // Assign key "key" with a string "Value";
 *   object["key"] = NestedKVStore::String("Value");
 *   // Assign key "arr" with a empty array;
 *   object["arr"] = NestedKVStore::Array();
 * \endcode
 */
class NestedKVStore {
 public:
  NestedKVStore() : ptr_{new NullValue} {}

  // integer
  NestedKVStore(IntegerValue number) : ptr_(new IntegerValue(number)) {}  // NOLINT
  NestedKVStore& operator=(IntegerValue number) {
    ptr_.reset(new IntegerValue(std::move(number)));
    return *this;
  }
  // number
  NestedKVStore(NumberValue number) : ptr_(new NumberValue(number)) {}  // NOLINT
  NestedKVStore& operator=(NumberValue number) {
    ptr_.reset(new NumberValue(std::move(number)));
    return *this;
  }
  // array
  NestedKVStore(ArrayValue list) :  // NOLINT
      ptr_(new ArrayValue(std::move(list))) {}
  NestedKVStore& operator=(ArrayValue array) {
    ptr_.reset(new ArrayValue(std::move(array)));
    return *this;
  }
  // object
  NestedKVStore(ObjectValue object) :  // NOLINT
      ptr_(new ObjectValue(std::move(object))) {}
  NestedKVStore& operator=(ObjectValue object) {
    ptr_.reset(new ObjectValue(std::move(object)));
    return *this;
  }
  // string
  NestedKVStore(StringValue str) :  // NOLINT
      ptr_(new StringValue(std::move(str))) {}
  NestedKVStore& operator=(StringValue str) {
    ptr_.reset(new StringValue(std::move(str)));
    return *this;
  }
  // bool
  NestedKVStore(BooleanValue boolean) :  // NOLINT
      ptr_(new BooleanValue(std::move(boolean))) {}
  NestedKVStore& operator=(BooleanValue boolean) {
    ptr_.reset(new BooleanValue(std::move(boolean)));
    return *this;
  }
  // null
  NestedKVStore(NullValue null) :  // NOLINT
      ptr_(new NullValue(std::move(null))) {}
  NestedKVStore& operator=(NullValue null) {
    ptr_.reset(new NullValue(std::move(null)));
    return *this;
  }

  // copy
  NestedKVStore(NestedKVStore const& other) = default;
  NestedKVStore& operator=(NestedKVStore const& other) = default;
  // move
  NestedKVStore(NestedKVStore&& other) : ptr_(std::move(other.ptr_)) {}
  NestedKVStore& operator=(NestedKVStore&& other) {
    ptr_ = std::move(other.ptr_);
    return *this;
  }

  /*! \brief Index NestedKVStore object with a std::string, used for object value. */
  NestedKVStore& operator[](std::string const & key) const { return (*ptr_)[key]; }
  /*! \brief Index NestedKVStore object with int, used for array value. */
  NestedKVStore& operator[](int ind)                 const { return (*ptr_)[ind]; }

  /*! \brief Append an NestedKVStore object to NestedKVStore lists */
  NestedKVStore& append(NestedKVStore val) const { return ptr_->append(val); }

  /*! \brief Return the reference to stored value. */
  Value& GetValue() { return *ptr_; }
  Value const& GetValue() const { return *ptr_; }

  bool operator==(NestedKVStore const& rhs) const {
    return *ptr_ == *(rhs.ptr_);
  }

 private:
  std::shared_ptr<Value> ptr_;
};

/*!
 * \brief Get value.
 *
 * \tparam T One of the serializer::value type.
 *
 * \param value
 * \return value with type T.
 */
template <typename T, typename U>
T& Get(U& value) {  // NOLINT
  auto& casted = *Cast<T>(&value.GetValue());
  return casted;
}

/*!
 * \brief Get value.
 *
 * \tparam T One of the serializer::value type.
 *
 * \param value
 * \return value with type T.
 */
template <typename T, typename U>
const T& Get(const U& value) {  // NOLINT
  const auto& casted = *Cast<const T>(&value.GetValue());
  return casted;
}

using Object = ObjectValue;
using Array = ArrayValue;
using Integer = IntegerValue;
using Number = NumberValue;
using Boolean = BooleanValue;
using String = StringValue;
using Null = NullValue;

// Helper utilities for handling dmlc::Parameter structure
// TODO(trivialfis): Try to integrate this better with dmlc-core
template <typename ParamType>
void InitParametersFromKVStore(
    const NestedKVStore& kv_store, const std::string& name, ParamType* parameters) {
  std::string key;
  auto& param_dict =
    serializer::Get<serializer::Object>(kv_store[name]).GetObject();
  std::map<std::string, std::string> tree_param_map;
  for (auto const& param_pair : param_dict) {
    std::string key = param_pair.first;
    std::string const& value =
        serializer::Get<serializer::String const>(param_pair.second).GetString();
    tree_param_map[key] = value;
  }
  parameters->Init(tree_param_map);
}

template <typename ParamType>
void SaveParametersToKVStore(
    NestedKVStore* p_kv_store, const ParamType& parameters, std::string name) {
  auto& kv_store = *p_kv_store;
  std::map<std::string, serializer::NestedKVStore> param_pairs;
  for (auto const& p : parameters.__DICT__()) {
    param_pairs[p.first] = serializer::String(p.second);
  }
  kv_store[name] = serializer::Object{param_pairs};
}

}      // namespace serializer
}      // namespace xgboost

#endif  // XGBOOST_COMMON_NESTED_KVSTORE_H_
