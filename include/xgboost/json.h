/*!
 * Copyright (c) by Contributors 2019
 */
#ifndef XGBOOST_JSON_H_
#define XGBOOST_JSON_H_

#include <xgboost/logging.h>
#include <xgboost/parameter.h>
#include <string>

#include <map>
#include <memory>
#include <vector>
#include <functional>
#include <utility>

namespace xgboost {

class Json;
class JsonReader;
class JsonWriter;

class Value {
 public:
  /*!\brief Simplified implementation of LLVM RTTI. */
  enum class ValueKind {
    String,
    Number,
    Integer,
    Object,  // std::map
    Array,   // std::vector
    Boolean,
    Null
  };

  explicit Value(ValueKind _kind) : kind_{_kind} {}

  ValueKind Type() const { return kind_; }
  virtual ~Value() = default;

  virtual void Save(JsonWriter* writer) = 0;

  virtual Json& operator[](std::string const & key) = 0;
  virtual Json& operator[](int ind) = 0;

  virtual bool operator==(Value const& rhs) const = 0;
  virtual Value& operator=(Value const& rhs) = 0;

  std::string TypeStr() const;

 private:
  ValueKind kind_;
};

template <typename T>
bool IsA(Value const* value) {
  return T::isClassOf(value);
}

template <typename T, typename U>
T* Cast(U* value) {
  if (IsA<T>(value)) {
    return dynamic_cast<T*>(value);
  } else {
    LOG(FATAL) << "Invalid cast, from " + value->TypeStr() + " to " + T().TypeStr();
  }
  return dynamic_cast<T*>(value);  // supress compiler warning.
}

class JsonString : public Value {
  std::string str_;
 public:
  JsonString() : Value(ValueKind::String) {}
  JsonString(std::string const& str) :  // NOLINT
      Value(ValueKind::String), str_{str} {}
  JsonString(std::string&& str) :  // NOLINT
      Value(ValueKind::String), str_{std::move(str)} {}

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  std::string const& getString() &&      { return str_; }
  std::string const& getString() const & { return str_; }
  std::string&       getString()       & { return str_; }

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  static bool isClassOf(Value const* value) {
    return value->Type() == ValueKind::String;
  }
};

class JsonArray : public Value {
  std::vector<Json> vec_;

 public:
  JsonArray() : Value(ValueKind::Array) {}
  JsonArray(std::vector<Json>&& arr) :  // NOLINT
      Value(ValueKind::Array), vec_{std::move(arr)} {}
  JsonArray(std::vector<Json> const& arr) :  // NOLINT
      Value(ValueKind::Array), vec_{arr} {}
  JsonArray(JsonArray const& that) = delete;
  JsonArray(JsonArray && that);

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  std::vector<Json> const& getArray() &&      { return vec_; }
  std::vector<Json> const& getArray() const & { return vec_; }
  std::vector<Json>&       getArray()       & { return vec_; }

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  static bool isClassOf(Value const* value) {
    return value->Type() == ValueKind::Array;
  }
};

class JsonObject : public Value {
  std::map<std::string, Json> object_;

 public:
  JsonObject() : Value(ValueKind::Object) {}
  JsonObject(std::map<std::string, Json>&& object);  // NOLINT
  JsonObject(JsonObject const& that) = delete;
  JsonObject(JsonObject && that);

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  std::map<std::string, Json> const& getObject() &&      { return object_; }
  std::map<std::string, Json> const& getObject() const & { return object_; }
  std::map<std::string, Json> &      getObject() &       { return object_; }

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  static bool isClassOf(Value const* value) {
    return value->Type() == ValueKind::Object;
  }
  virtual ~JsonObject() = default;
};

class JsonNumber : public Value {
 public:
  using Float = float;

 private:
  Float number_;

 public:
  JsonNumber() : Value(ValueKind::Number) {}
  template <typename FloatT,
            typename std::enable_if<std::is_same<FloatT, Float>::value>::type* = nullptr>
  JsonNumber(FloatT value) : Value(ValueKind::Number) {  // NOLINT
    number_ = value;
  }
  template <typename FloatT,
            typename std::enable_if<std::is_same<FloatT, double>::value>::type* = nullptr>
  JsonNumber(FloatT value) : Value{ValueKind::Number},  // NOLINT
                             number_{static_cast<Float>(value)} {}

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  Float const& getNumber() &&      { return number_; }
  Float const& getNumber() const & { return number_; }
  Float&       getNumber()       & { return number_; }


  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  static bool isClassOf(Value const* value) {
    return value->Type() == ValueKind::Number;
  }
};

class JsonInteger : public Value {
 public:
  using Int = int64_t;

 private:
  Int integer_;

 public:
  JsonInteger() : Value(ValueKind::Integer), integer_{0} {}  // NOLINT
  template <typename IntT,
            typename std::enable_if<std::is_same<IntT, Int>::value>::type* = nullptr>
  JsonInteger(IntT value) : Value(ValueKind::Integer), integer_{value} {} // NOLINT
  template <typename IntT,
            typename std::enable_if<std::is_same<IntT, size_t>::value>::type* = nullptr>
  JsonInteger(IntT value) : Value(ValueKind::Integer),  // NOLINT
                            integer_{static_cast<Int>(value)} {}
  template <typename IntT,
            typename std::enable_if<std::is_same<IntT, int32_t>::value>::type* = nullptr>
  JsonInteger(IntT value) : Value(ValueKind::Integer),  // NOLINT
                            integer_{static_cast<Int>(value)} {}
  template <typename IntT,
            typename std::enable_if<
                std::is_same<IntT, uint32_t>::value &&
                !std::is_same<std::size_t, uint32_t>::value>::type * = nullptr>
  JsonInteger(IntT value)  // NOLINT
      : Value(ValueKind::Integer),
        integer_{static_cast<Int>(value)} {}

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  Int const& getInteger() &&      { return integer_; }
  Int const& getInteger() const & { return integer_; }
  Int& getInteger() &             { return integer_; }
  void Save(JsonWriter* writer) override;

  static bool isClassOf(Value const* value) {
    return value->Type() == ValueKind::Integer;
  }
};

class JsonNull : public Value {
 public:
  JsonNull() : Value(ValueKind::Null) {}
  JsonNull(std::nullptr_t) : Value(ValueKind::Null) {}  // NOLINT

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  static bool isClassOf(Value const* value) {
    return value->Type() == ValueKind::Null;
  }
};

/*! \brief Describes both true and false. */
class JsonBoolean : public Value {
  bool boolean_;

 public:
  JsonBoolean() : Value(ValueKind::Boolean) {}  // NOLINT
  // Ambigious with JsonNumber.
  template <typename Bool,
            typename std::enable_if<
              std::is_same<Bool, bool>::value ||
              std::is_same<Bool, bool const>::value>::type* = nullptr>
  JsonBoolean(Bool value) :  // NOLINT
      Value(ValueKind::Boolean), boolean_{value} {}

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  bool const& getBoolean() &&      { return boolean_; }
  bool const& getBoolean() const & { return boolean_; }
  bool&       getBoolean()       & { return boolean_; }

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  static bool isClassOf(Value const* value) {
    return value->Type() == ValueKind::Boolean;
  }
};

struct StringView {
  using CharT = char;  // unsigned char
  CharT const* str_;
  size_t size_;

 public:
  StringView() = default;
  StringView(CharT const* str, size_t size) : str_{str}, size_{size} {}

  CharT const& operator[](size_t p) const { return str_[p]; }
  CharT const& at(size_t p) const {  // NOLINT
    CHECK_LT(p, size_);
    return str_[p];
  }
  size_t size() const { return size_; }  // NOLINT
  // Copies a portion of string.  Since we don't have std::from_chars and friends here, so
  // copying substring is necessary for appending `\0`.  It's not too bad since string by
  // default has small vector optimization, which is enabled by most if not all modern
  // compilers for numeric values.
  std::string substr(size_t beg, size_t n) const {  // NOLINT
    CHECK_LE(beg, size_);
    return std::string {str_ + beg, n < (size_ - beg) ? n : (size_ - beg)};
  }
  char const* c_str() const { return str_; }  // NOLINT
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
 *   Json object { Object() };
 *   // Assign key "key" with a JSON string "Value";
 *   object["key"] = String("Value");
 *   // Assign key "arr" with a empty JSON Array;
 *   object["arr"] = Array();
 * \endcode
 */
class Json {
  friend JsonWriter;

 public:
  /*! \brief Load a Json object from string. */
  static Json Load(StringView str);
  /*! \brief Pass your own JsonReader. */
  static Json Load(JsonReader* reader);
  /*! \brief Dump json into stream. */
  static void Dump(Json json, std::ostream* stream,
                   bool pretty = ConsoleLogger::ShouldLog(
                       ConsoleLogger::LogVerbosity::kDebug));
  static void Dump(Json json, std::string* out,
                   bool pretty = ConsoleLogger::ShouldLog(
                       ConsoleLogger::LogVerbosity::kDebug));

  Json() : ptr_{new JsonNull} {}

  // number
  explicit Json(JsonNumber number) : ptr_{new JsonNumber(number)} {}
  Json& operator=(JsonNumber number) {
    ptr_.reset(new JsonNumber(std::move(number)));
    return *this;
  }

  // integer
  explicit Json(JsonInteger integer) : ptr_{new JsonInteger(integer)} {}
  Json& operator=(JsonInteger integer) {
    ptr_.reset(new JsonInteger(std::move(integer)));
    return *this;
  }

  // array
  explicit Json(JsonArray list) :
      ptr_ {new JsonArray(std::move(list))} {}
  Json& operator=(JsonArray array) {
    ptr_.reset(new JsonArray(std::move(array)));
    return *this;
  }

  // object
  explicit Json(JsonObject object) :
      ptr_{new JsonObject(std::move(object))} {}
  Json& operator=(JsonObject object) {
    ptr_.reset(new JsonObject(std::move(object)));
    return *this;
  }
  // string
  explicit Json(JsonString str) :
      ptr_{new JsonString(std::move(str))} {}
  Json& operator=(JsonString str) {
    ptr_.reset(new JsonString(std::move(str)));
    return *this;
  }
  // bool
  explicit Json(JsonBoolean boolean) :
      ptr_{new JsonBoolean(std::move(boolean))} {}
  Json& operator=(JsonBoolean boolean) {
    ptr_.reset(new JsonBoolean(std::move(boolean)));
    return *this;
  }
  // null
  explicit Json(JsonNull null) :
      ptr_{new JsonNull(std::move(null))} {}
  Json& operator=(JsonNull null) {
    ptr_.reset(new JsonNull(std::move(null)));
    return *this;
  }

  // copy
  Json(Json const& other) : ptr_{other.ptr_} {}
  Json& operator=(Json const& other);
  // move
  Json(Json&& other) : ptr_{std::move(other.ptr_)} {}
  Json& operator=(Json&& other) {
    ptr_ = std::move(other.ptr_);
    return *this;
  }

  /*! \brief Index Json object with a std::string, used for Json Object. */
  Json& operator[](std::string const & key) const { return (*ptr_)[key]; }
  /*! \brief Index Json object with int, used for Json Array. */
  Json& operator[](int ind)                 const { return (*ptr_)[ind]; }

  /*! \Brief Return the reference to stored Json value. */
  Value const& GetValue() const & { return *ptr_; }
  Value const& GetValue() &&      { return *ptr_; }
  Value&       GetValue() &       { return *ptr_; }

  bool operator==(Json const& rhs) const {
    return *ptr_ == *(rhs.ptr_);
  }

  friend std::ostream& operator<<(std::ostream& os, Json const& j) {
    std::string str;
    Json::Dump(j, &str);
    os << str;
    return os;
  }

 private:
  std::shared_ptr<Value> ptr_;
};

template <typename T>
bool IsA(Json const j) {
  auto const& v = j.GetValue();
  return IsA<T>(&v);
}

namespace detail {

// Number
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonNumber>::value>::type* = nullptr>
JsonNumber::Float& GetImpl(T& val) {  // NOLINT
  return val.getNumber();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonNumber const>::value>::type* = nullptr>
JsonNumber::Float const& GetImpl(T& val) {  // NOLINT
  return val.getNumber();
}

// Integer
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonInteger>::value>::type* = nullptr>
JsonInteger::Int& GetImpl(T& val) {  // NOLINT
  return val.getInteger();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonInteger const>::value>::type* = nullptr>
JsonInteger::Int const& GetImpl(T& val) {  // NOLINT
  return val.getInteger();
}

// String
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonString>::value>::type* = nullptr>
std::string& GetImpl(T& val) {  // NOLINT
  return val.getString();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonString const>::value>::type* = nullptr>
std::string const& GetImpl(T& val) {  // NOLINT
  return val.getString();
}

// Boolean
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonBoolean>::value>::type* = nullptr>
bool& GetImpl(T& val) {  // NOLINT
  return val.getBoolean();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonBoolean const>::value>::type* = nullptr>
bool const& GetImpl(T& val) {  // NOLINT
  return val.getBoolean();
}

// Array
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonArray>::value>::type* = nullptr>
std::vector<Json>& GetImpl(T& val) {  // NOLINT
  return val.getArray();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonArray const>::value>::type* = nullptr>
std::vector<Json> const& GetImpl(T& val) {  // NOLINT
  return val.getArray();
}

// Object
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonObject>::value>::type* = nullptr>
std::map<std::string, Json>& GetImpl(T& val) {  // NOLINT
  return val.getObject();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonObject const>::value>::type* = nullptr>
std::map<std::string, Json> const& GetImpl(T& val) {  // NOLINT
  return val.getObject();
}

}  // namespace detail

/*!
 * \brief Get Json value.
 *
 * \tparam T One of the Json value type.
 *
 * \param json
 * \return Value contained in Json object of type T.
 */
template <typename T, typename U>
auto get(U& json) -> decltype(detail::GetImpl(*Cast<T>(&json.GetValue())))& { // NOLINT
  auto& value = *Cast<T>(&json.GetValue());
  return detail::GetImpl(value);
}

using Object  = JsonObject;
using Array   = JsonArray;
using Number  = JsonNumber;
using Integer = JsonInteger;
using Boolean = JsonBoolean;
using String  = JsonString;
using Null    = JsonNull;

// Utils tailored for XGBoost.

template <typename Parameter>
Object toJson(Parameter const& param) {
  Object obj;
  for (auto const& kv : param.__DICT__()) {
    obj[kv.first] = kv.second;
  }
  return obj;
}

template <typename Parameter>
void fromJson(Json const& obj, Parameter* param) {
  auto const& j_param = get<Object const>(obj);
  std::map<std::string, std::string> m;
  for (auto const& kv : j_param) {
    m[kv.first] = get<String const>(kv.second);
  }
  param->UpdateAllowUnknown(m);
}
}  // namespace xgboost
#endif  // XGBOOST_JSON_H_
