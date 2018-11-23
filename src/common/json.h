/*!
 * Copyright 2018 by Contributors
 * \file json.h
 * \brief Simple implementation of JSON.
 */
#ifndef XGBOOST_COMMON_JSON_H_
#define XGBOOST_COMMON_JSON_H_

#include <iostream>
#include <istream>
#include <string>
#include <sstream>

#include <map>
#include <memory>
#include <vector>

namespace xgboost {
namespace json {

class Json;
class JsonWriter;

class Value {
 protected:
  /*!\brief Simplified implementation of LLVM RTTI. */
  enum class ValueKind {
    kString,
    kNumber,
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

  virtual void Save(JsonWriter* stream) = 0;

  virtual Json& operator[](std::string const & key) = 0;
  virtual Json& operator[](int ind) = 0;

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

class JsonString : public Value {
  std::string str_;
 public:
  JsonString() : Value(ValueKind::kString) {}
  JsonString(std::string const& str) :  // NOLINT
      Value(ValueKind::kString), str_(str) {}
  JsonString(std::string&& str) :  // NOLINT
      Value(ValueKind::kString), str_(std::move(str)) {}

  void Save(JsonWriter* stream) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  bool operator==(Value const& rhs) const override;

  std::string const& GetString() const { return str_; }
  std::string & GetString() { return str_;}

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kString;
  }
};

class JsonArray : public Value {
  std::vector<Json> vec_;
 public:
  JsonArray();
  JsonArray(std::vector<Json>&& arr);        // NOLINT
  JsonArray(std::vector<Json> const& arr);   // NOLINT

  void Save(JsonWriter* stream) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  std::vector<Json> const& GetArray() const { return vec_; }
  std::vector<Json> & GetArray() { return vec_; }

  bool operator==(Value const& rhs) const override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kArray;
  }
};

class JsonObject : public Value {
  std::map<std::string, Json> object_;

 public:
  JsonObject();                                           // NOLINT
  JsonObject(std::map<std::string, Json>&& object);       // NOLINT
  JsonObject(std::map<std::string, Json> const& object);  // NOLINT

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  bool operator==(Value const& rhs) const override;

  std::map<std::string, Json> const& GetObject() const { return object_; }
  std::map<std::string, Json> &      GetObject()       { return object_; }

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kObject;
  }
};

class JsonNumber : public Value {
  double number_;

 public:
  JsonNumber() : Value(ValueKind::kNumber), number_{0} {}
  JsonNumber(double value);  // NOLINT

  void Save(JsonWriter* stream) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  bool operator==(Value const& rhs) const override;

  int    GetInteger() const;
  double GetDouble()  const;
  float  GetFloat()   const;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kNumber;
  }
};

class JsonNull : public Value {
 public:
  JsonNull() : Value(ValueKind::kNull) {}

  void Save(JsonWriter* stream) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  bool operator==(Value const& rhs) const override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kNull;
  }
};

/*! \brief Describes both true and false. */
class JsonBoolean : public Value {
  bool boolean_;
 public:
  JsonBoolean() : Value(ValueKind::kBoolean) {}
  // Ambigious with JsonNumber.
  template <typename Bool,
            typename std::enable_if<
              std::is_same<Bool, bool>::value ||
              std::is_same<Bool, bool const>::value>::type* = nullptr>
  JsonBoolean(Bool value) :  // NOLINT
      Value(ValueKind::kBoolean), boolean_(value) {}

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  bool operator==(Value const& rhs) const override;

  bool GetBoolean() const { return boolean_; }

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kBoolean;
  }
};

/*!
 * \brief Data structure representing JSON format.
 *
 * Limitation:  UTF-8 is not properly supported.  Code points above ASCII are
 *              invalid.
 *
 * Examples:
 *
 * \code
 *   // Create a JSON object.
 *   json::Json object = json::Object();
 *   // Assign key "key" with a JSON string "Value";
 *   object["key"] = Json::String("Value");
 *   // Assign key "arr" with a empty JSON Array;
 *   object["arr"] = Json::Array();
 * \endcode
 */
class Json {
  friend JsonWriter;
  void Save(JsonWriter* writer) {
    this->ptr_->Save(writer);
  }

 public:
  /*! \brief Load a Json file from stream. */
  static Json Load(std::istream* stream);
  /*! \brief Dump json into stream. */
  static void Dump(Json json, std::ostream* stream);

  Json() : ptr_{new JsonNull} {}

  // number
  Json(JsonNumber number) : ptr_(new JsonNumber(number)) {}  // NOLINT
  Json& operator=(JsonNumber number) {
    ptr_.reset(new JsonNumber(std::move(number)));
    return *this;
  }
  // array
  Json(JsonArray list) :  // NOLINT
      ptr_(new JsonArray(std::move(list))) {}
  Json& operator=(JsonArray array) {
    ptr_.reset(new JsonArray(std::move(array)));
    return *this;
  }
  // object
  Json(JsonObject object) :  // NOLINT
      ptr_(new JsonObject(std::move(object))) {}
  Json& operator=(JsonObject object) {
    ptr_.reset(new JsonObject(std::move(object)));
    return *this;
  }
  // string
  Json(JsonString str) :  // NOLINT
      ptr_(new JsonString(std::move(str))) {}
  Json& operator=(JsonString str) {
    ptr_.reset(new JsonString(std::move(str)));
    return *this;
  }
  // bool
  Json(JsonBoolean boolean) :  // NOLINT
      ptr_(new JsonBoolean(std::move(boolean))) {}
  Json& operator=(JsonBoolean boolean) {
    ptr_.reset(new JsonBoolean(std::move(boolean)));
    return *this;
  }
  // null
  Json(JsonNull null) :  // NOLINT
      ptr_(new JsonNull(std::move(null))) {}
  Json& operator=(JsonNull null) {
    ptr_.reset(new JsonNull(std::move(null)));
    return *this;
  }

  // copy
  Json(Json const& other) = default;
  Json& operator=(Json const& other) = default;
  // move
  Json(Json&& other) : ptr_(std::move(other.ptr_)) {}
  Json& operator=(Json&& other) {
    ptr_ = std::move(other.ptr_);
    return *this;
  }

  /*! \brief Index Json object with a std::string, used for Json Object. */
  Json& operator[](std::string const & key) const { return (*ptr_)[key]; }
  /*! \brief Index Json object with int, used for Json Array. */
  Json& operator[](int ind)                 const { return (*ptr_)[ind]; }

  /*! \brief Return the reference to stored Json value. */
  Value& GetValue() { return *ptr_; }
  Value const& GetValue() const { return *ptr_; }

  bool operator==(Json const& rhs) const {
    return *ptr_ == *(rhs.ptr_);
  }

 private:
  std::shared_ptr<Value> ptr_;
};

/*!
 * \brief Get Json value.
 *
 * \tparam T One of the Json value type.
 *
 * \param json
 * \return Json value with type T.
 */
template <typename T, typename U>
T& Get(U& json) {  // NOLINT
  auto& value = *Cast<T>(&json.GetValue());
  return value;
}

using Object = JsonObject;
using Array = JsonArray;
using Number = JsonNumber;
using Boolean = JsonBoolean;
using String = JsonString;
using Null = JsonNull;

// Helper utilities for handling dmlc::Parameters
// TODO(trivialfis): Try to integrate this better with dmlc-core
template <typename P>
void InitParametersFromJson(
    Json const& r_json, std::string const& name, P *parameters) {
  std::string key;
  auto& param_json_map =
      json::Get<json::Object>(r_json[name]).GetObject();
  std::map<std::string, std::string> tree_param_map;
  for (auto const& param_pair : param_json_map) {
    std::string key = param_pair.first;
    std::string const& value =
        json::Get<json::String const>(param_pair.second).GetString();
    tree_param_map[key] = value;
  }
  parameters->Init(tree_param_map);
}

template <typename P>
void SaveParametersToJson(
    Json* p_json, P const& parameters, std::string name) {
    auto& r_json = *p_json;
    std::map<std::string, json::Json> param_pairs;
    for (auto const& p : parameters.__DICT__()) {
      param_pairs[p.first] = json::String(p.second);
    }
    r_json[name] = json::Object{param_pairs};
}

}      // namespace json
}      // namespace xgboost

#endif  // XGBOOST_COMMON_JSON_H_
