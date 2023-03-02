/**
 * Copyright 2019-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_JSON_H_
#define XGBOOST_JSON_H_

#include <xgboost/intrusive_ptr.h>
#include <xgboost/logging.h>
#include <xgboost/parameter.h>
#include <xgboost/string_view.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>  // std::enable_if,std::enable_if_t
#include <utility>
#include <vector>

namespace xgboost {

class Json;
class JsonReader;
class JsonWriter;

class Value {
 private:
  mutable class IntrusivePtrCell ref_;
  friend IntrusivePtrCell &
  IntrusivePtrRefCount(xgboost::Value const *t) noexcept {
    return t->ref_;
  }

 public:
  /*!\brief Simplified implementation of LLVM RTTI. */
  enum class ValueKind {
    kString,
    kNumber,
    kInteger,
    kObject,  // std::map
    kArray,   // std::vector
    kBoolean,
    kNull,
    // typed array for ubjson
    kNumberArray,
    kU8Array,
    kI32Array,
    kI64Array
  };

  explicit Value(ValueKind _kind) : kind_{_kind} {}

  ValueKind Type() const { return kind_; }
  virtual ~Value() = default;

  virtual void Save(JsonWriter* writer) const = 0;

  virtual Json& operator[](std::string const& key);
  virtual Json& operator[](int ind);

  virtual bool operator==(Value const& rhs) const = 0;
#if !defined(__APPLE__)
  virtual Value& operator=(Value const& rhs) = delete;
#endif  // !defined(__APPLE__)

  std::string TypeStr() const;

 private:
  ValueKind kind_;
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
    LOG(FATAL) << "Invalid cast, from " + value->TypeStr() + " to " + T().TypeStr();
  }
  return dynamic_cast<T*>(value);  // suppress compiler warning.
}

class JsonString : public Value {
  std::string str_;

 public:
  JsonString() : Value(ValueKind::kString) {}
  JsonString(std::string const& str) :  // NOLINT
      Value(ValueKind::kString), str_{str} {}
  JsonString(std::string&& str) noexcept :  // NOLINT
      Value(ValueKind::kString), str_{std::forward<std::string>(str)} {}
  JsonString(JsonString&& str) noexcept : Value(ValueKind::kString) {  // NOLINT
    std::swap(str.str_, this->str_);
  }

  void Save(JsonWriter* writer) const override;

  std::string const& GetString() &&      { return str_; }
  std::string const& GetString() const & { return str_; }
  std::string&       GetString()       & { return str_; }

  bool operator==(Value const& rhs) const override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kString;
  }
};

class JsonArray : public Value {
  std::vector<Json> vec_;

 public:
  JsonArray() : Value(ValueKind::kArray) {}
  JsonArray(std::vector<Json>&& arr) noexcept  // NOLINT
      : Value(ValueKind::kArray), vec_{std::forward<std::vector<Json>>(arr)} {}
  JsonArray(std::vector<Json> const& arr) :  // NOLINT
      Value(ValueKind::kArray), vec_{arr} {}
  JsonArray(JsonArray const& that) = delete;
  JsonArray(JsonArray && that) noexcept;

  void Save(JsonWriter* writer) const override;

  Json& operator[](int ind) override { return vec_.at(ind); }
  // silent the partial oveeridden warning
  Json& operator[](std::string const& key) override { return Value::operator[](key); }

  std::vector<Json> const& GetArray() &&      { return vec_; }
  std::vector<Json> const& GetArray() const & { return vec_; }
  std::vector<Json>&       GetArray()       & { return vec_; }

  bool operator==(Value const& rhs) const override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kArray;
  }
};

/**
 * \brief Typed array for Universal Binary JSON.
 *
 * \tparam T The underlying primitive type.
 * \tparam kind Value kind defined by JSON type.
 */
template <typename T, Value::ValueKind kind>
class JsonTypedArray : public Value {
  std::vector<T> vec_;

 public:
  using Type = T;

  JsonTypedArray() : Value(kind) {}
  explicit JsonTypedArray(size_t n) : Value(kind) { vec_.resize(n); }
  JsonTypedArray(JsonTypedArray&& that) noexcept : Value{kind}, vec_{std::move(that.vec_)} {}

  bool operator==(Value const& rhs) const override;

  void Set(size_t i, T v) { vec_[i] = v; }
  size_t Size() const { return vec_.size(); }

  void Save(JsonWriter* writer) const override;

  std::vector<T> const& GetArray() && { return vec_; }
  std::vector<T> const& GetArray() const& { return vec_; }
  std::vector<T>& GetArray() & { return vec_; }

  static bool IsClassOf(Value const* value) { return value->Type() == kind; }
};

/**
 * \brief Typed UBJSON array for 32-bit floating point.
 */
using F32Array = JsonTypedArray<float, Value::ValueKind::kNumberArray>;
/**
 * \brief Typed UBJSON array for uint8_t.
 */
using U8Array = JsonTypedArray<uint8_t, Value::ValueKind::kU8Array>;
/**
 * \brief Typed UBJSON array for int32_t.
 */
using I32Array = JsonTypedArray<int32_t, Value::ValueKind::kI32Array>;
/**
 * \brief Typed UBJSON array for int64_t.
 */
using I64Array = JsonTypedArray<int64_t, Value::ValueKind::kI64Array>;

class JsonObject : public Value {
 public:
  using Map = std::map<std::string, Json, std::less<>>;

 private:
  Map object_;

 public:
  JsonObject() : Value(ValueKind::kObject) {}
  JsonObject(Map&& object) noexcept;  // NOLINT
  JsonObject(JsonObject const& that) = delete;
  JsonObject(JsonObject&& that) noexcept;

  void Save(JsonWriter* writer) const override;

  // silent the partial oveeridden warning
  Json& operator[](int ind) override { return Value::operator[](ind); }
  Json& operator[](std::string const& key) override { return object_[key]; }

  Map const& GetObject() && { return object_; }
  Map const& GetObject() const& { return object_; }
  Map& GetObject() & { return object_; }

  bool operator==(Value const& rhs) const override;

  static bool IsClassOf(Value const* value) { return value->Type() == ValueKind::kObject; }
  ~JsonObject() override = default;
};

class JsonNumber : public Value {
 public:
  using Float = float;

 private:
  Float number_ { 0 };

 public:
  JsonNumber() : Value(ValueKind::kNumber) {}
  template <typename FloatT,
            typename std::enable_if<std::is_same<FloatT, Float>::value>::type* = nullptr>
  JsonNumber(FloatT value) : Value(ValueKind::kNumber) {  // NOLINT
    number_ = value;
  }
  template <typename FloatT,
            typename std::enable_if<std::is_same<FloatT, double>::value>::type* = nullptr>
  JsonNumber(FloatT value) : Value{ValueKind::kNumber},  // NOLINT
                             number_{static_cast<Float>(value)} {}
  JsonNumber(JsonNumber const& that) = delete;
  JsonNumber(JsonNumber&& that) noexcept : Value{ValueKind::kNumber}, number_{that.number_} {}

  void Save(JsonWriter* writer) const override;

  Float const& GetNumber() &&      { return number_; }
  Float const& GetNumber() const & { return number_; }
  Float&       GetNumber()       & { return number_; }

  bool operator==(Value const& rhs) const override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kNumber;
  }
};

class JsonInteger : public Value {
 public:
  using Int = int64_t;

 private:
  Int integer_ {0};

 public:
  JsonInteger() : Value(ValueKind::kInteger) {}  // NOLINT
  template <typename IntT,
            typename std::enable_if<std::is_same<IntT, Int>::value>::type* = nullptr>
  JsonInteger(IntT value) : Value(ValueKind::kInteger), integer_{value} {} // NOLINT
  template <typename IntT,
            typename std::enable_if<std::is_same<IntT, size_t>::value>::type* = nullptr>
  JsonInteger(IntT value) : Value(ValueKind::kInteger),  // NOLINT
                            integer_{static_cast<Int>(value)} {}
  template <typename IntT,
            typename std::enable_if<std::is_same<IntT, int32_t>::value>::type* = nullptr>
  JsonInteger(IntT value) : Value(ValueKind::kInteger),  // NOLINT
                            integer_{static_cast<Int>(value)} {}
  template <typename IntT,
            typename std::enable_if<
                std::is_same<IntT, uint32_t>::value &&
                !std::is_same<std::size_t, uint32_t>::value>::type * = nullptr>
  JsonInteger(IntT value)  // NOLINT
      : Value(ValueKind::kInteger),
        integer_{static_cast<Int>(value)} {}

  JsonInteger(JsonInteger &&that) noexcept
      : Value{ValueKind::kInteger}, integer_{that.integer_} {}

  bool operator==(Value const& rhs) const override;

  Int const& GetInteger() &&      { return integer_; }
  Int const& GetInteger() const & { return integer_; }
  Int& GetInteger() &             { return integer_; }
  void Save(JsonWriter* writer) const override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kInteger;
  }
};

class JsonNull : public Value {
 public:
  JsonNull() : Value(ValueKind::kNull) {}
  JsonNull(std::nullptr_t) : Value(ValueKind::kNull) {}  // NOLINT
  JsonNull(JsonNull&&) noexcept : Value(ValueKind::kNull) {}

  void Save(JsonWriter* writer) const override;

  bool operator==(Value const& rhs) const override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kNull;
  }
};

/*! \brief Describes both true and false. */
class JsonBoolean : public Value {
  bool boolean_ = false;

 public:
  JsonBoolean() : Value(ValueKind::kBoolean) {}  // NOLINT
  // Ambigious with JsonNumber.
  template <typename Bool,
            typename std::enable_if<
              std::is_same<Bool, bool>::value ||
              std::is_same<Bool, bool const>::value>::type* = nullptr>
  JsonBoolean(Bool value) :  // NOLINT
      Value(ValueKind::kBoolean), boolean_{value} {}
  JsonBoolean(JsonBoolean&& value) noexcept:  // NOLINT
      Value(ValueKind::kBoolean), boolean_{value.boolean_} {}

  void Save(JsonWriter* writer) const override;

  bool const& GetBoolean() &&      { return boolean_; }
  bool const& GetBoolean() const & { return boolean_; }
  bool&       GetBoolean()       & { return boolean_; }

  bool operator==(Value const& rhs) const override;

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
 *   Json object { Object() };
 *   // Assign key "key" with a JSON string "Value";
 *   object["key"] = String("Value");
 *   // Assign key "arr" with a empty JSON Array;
 *   object["arr"] = Array();
 * \endcode
 */
class Json {
 public:
  /**
   *  \brief Decode the JSON object.  Optional parameter mode for choosing between text
   *         and binary (ubjson) input.
   */
  static Json Load(StringView str, std::ios::openmode mode = std::ios::in);
  /*! \brief Pass your own JsonReader. */
  static Json Load(JsonReader* reader);
  /**
   *  \brief Encode the JSON object.  Optional parameter mode for choosing between text
   *         and binary (ubjson) output.
   */
  static void Dump(Json json, std::string* out, std::ios::openmode mode = std::ios::out);
  static void Dump(Json json, std::vector<char>* out, std::ios::openmode mode = std::ios::out);
  /*! \brief Use your own JsonWriter. */
  static void Dump(Json json, JsonWriter* writer);

  Json() = default;

  // number
  explicit Json(JsonNumber number) : ptr_{new JsonNumber(std::move(number))} {}
  Json& operator=(JsonNumber number) {
    ptr_.reset(new JsonNumber(std::move(number)));
    return *this;
  }
  // integer
  explicit Json(JsonInteger integer) : ptr_{new JsonInteger(std::move(integer))} {}
  Json& operator=(JsonInteger integer) {
    ptr_.reset(new JsonInteger(std::move(integer)));
    return *this;
  }
  // array
  explicit Json(JsonArray&& list) : ptr_{new JsonArray(std::forward<JsonArray>(list))} {}
  Json& operator=(JsonArray&& array) {
    ptr_.reset(new JsonArray(std::forward<JsonArray>(array)));
    return *this;
  }
  // typed array
  template <typename T, Value::ValueKind kind>
  explicit Json(JsonTypedArray<T, kind>&& list)
      : ptr_{new JsonTypedArray<T, kind>(std::forward<JsonTypedArray<T, kind>>(list))} {}
  template <typename T, Value::ValueKind kind>
  Json& operator=(JsonTypedArray<T, kind>&& array) {
    ptr_.reset(new JsonTypedArray<T, kind>(std::forward<JsonTypedArray<T, kind>>(array)));
    return *this;
  }
  // object
  explicit Json(JsonObject&& object) : ptr_{new JsonObject(std::forward<JsonObject>(object))} {}
  Json& operator=(JsonObject&& object) {
    ptr_.reset(new JsonObject(std::forward<JsonObject>(object)));
    return *this;
  }
  // string
  explicit Json(JsonString&& str) : ptr_{new JsonString(std::forward<JsonString>(str))} {}
  Json& operator=(JsonString&& str) {
    ptr_.reset(new JsonString(std::forward<JsonString>(str)));
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
  Json(Json const& other) = default;
  Json& operator=(Json const& other) = default;
  // move
  Json(Json &&other) noexcept { std::swap(this->ptr_, other.ptr_); }
  Json &operator=(Json &&other) noexcept {
    std::swap(this->ptr_, other.ptr_);
    return *this;
  }

  /*! \brief Index Json object with a std::string, used for Json Object. */
  Json& operator[](std::string const & key) const { return (*ptr_)[key]; }
  /*! \brief Index Json object with int, used for Json Array. */
  Json& operator[](int ind)                 const { return (*ptr_)[ind]; }

  /*! \brief Return the reference to stored Json value. */
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

  IntrusivePtr<Value> const& Ptr() const { return ptr_; }

 private:
  IntrusivePtr<Value> ptr_{new JsonNull};
};

/**
 * \brief Check whether a Json object has specific type.
 *
 * \code
 *   Json json {Array{}};
 *   bool is_array = IsA<Array>(json);
 *   CHECK(is_array);
 * \endcode
 */
template <typename T>
bool IsA(Json const& j) {
  auto const& v = j.GetValue();
  return IsA<T>(&v);
}

namespace detail {
// Number
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonNumber>::value>::type* = nullptr>
JsonNumber::Float& GetImpl(T& val) {  // NOLINT
  return val.GetNumber();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonNumber const>::value>::type* = nullptr>
JsonNumber::Float const& GetImpl(T& val) {  // NOLINT
  return val.GetNumber();
}

// Integer
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonInteger>::value>::type* = nullptr>
JsonInteger::Int& GetImpl(T& val) {  // NOLINT
  return val.GetInteger();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonInteger const>::value>::type* = nullptr>
JsonInteger::Int const& GetImpl(T& val) {  // NOLINT
  return val.GetInteger();
}

// String
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonString>::value>::type* = nullptr>
std::string& GetImpl(T& val) {  // NOLINT
  return val.GetString();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonString const>::value>::type* = nullptr>
std::string const& GetImpl(T& val) {  // NOLINT
  return val.GetString();
}

// Boolean
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonBoolean>::value>::type* = nullptr>
bool& GetImpl(T& val) {  // NOLINT
  return val.GetBoolean();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonBoolean const>::value>::type* = nullptr>
bool const& GetImpl(T& val) {  // NOLINT
  return val.GetBoolean();
}

// Array
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonArray>::value>::type* = nullptr>
std::vector<Json>& GetImpl(T& val) {  // NOLINT
  return val.GetArray();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonArray const>::value>::type* = nullptr>
std::vector<Json> const& GetImpl(T& val) {  // NOLINT
  return val.GetArray();
}

// Typed Array
template <typename T, Value::ValueKind kind>
std::vector<T>& GetImpl(JsonTypedArray<T, kind>& val) {  // NOLINT
  return val.GetArray();
}
template <typename T, Value::ValueKind kind>
std::vector<T> const& GetImpl(JsonTypedArray<T, kind> const& val) {
  return val.GetArray();
}

// Object
template <typename T, typename std::enable_if<std::is_same<T, JsonObject>::value>::type* = nullptr>
JsonObject::Map& GetImpl(T& val) {  // NOLINT
  return val.GetObject();
}
template <typename T,
          typename std::enable_if<std::is_same<T, JsonObject const>::value>::type* = nullptr>
JsonObject::Map const& GetImpl(T& val) {  // NOLINT
  return val.GetObject();
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
namespace detail {
template <typename Head>
bool TypeCheckImpl(Json const& value) {
  return IsA<Head>(value);
}

template <typename Head, typename... JT>
std::enable_if_t<sizeof...(JT) != 0, bool> TypeCheckImpl(Json const& value) {
  return IsA<Head>(value) || TypeCheckImpl<JT...>(value);
}

template <typename Head>
std::string TypeCheckError() {
  return "`" + Head{}.TypeStr() + "`";
}

template <typename Head, typename... JT>
std::enable_if_t<sizeof...(JT) != 0, std::string> TypeCheckError() {
  return "`" + Head{}.TypeStr() + "`, " + TypeCheckError<JT...>();
}
}  // namespace detail

/**
 * \brief Type check for JSON-based parameters
 *
 * \tparam JT    Expected JSON types.
 * \param  value Value to be checked.
 */
template <typename... JT>
void TypeCheck(Json const& value, StringView name) {
  if (!detail::TypeCheckImpl<JT...>(value)) {
    LOG(FATAL) << "Invalid type for: `" << name << "`, expecting one of the: {`"
               << detail::TypeCheckError<JT...>() << "}, got: `" << value.GetValue().TypeStr()
               << "`";
  }
}

/**
 * \brief Convert XGBoost parameter to JSON object.
 *
 * \tparam Parameter An instantiation of XGBoostParameter
 *
 * \param param Input parameter
 *
 * \return JSON object representing the input parameter
 */
template <typename Parameter>
Object ToJson(Parameter const& param) {
  Object obj;
  for (auto const& kv : param.__DICT__()) {
    obj[kv.first] = kv.second;
  }
  return obj;
}

/**
 * \brief Load a XGBoost parameter from a JSON object.
 *
 * \tparam Parameter An instantiation of XGBoostParameter
 *
 * \param obj JSON object representing the parameter.
 * \param param Output parameter.
 *
 * \return Unknown arguments in the JSON object.
 */
template <typename Parameter>
Args FromJson(Json const& obj, Parameter* param) {
  auto const& j_param = get<Object const>(obj);
  std::map<std::string, std::string> m;
  for (auto const& kv : j_param) {
    m[kv.first] = get<String const>(kv.second);
  }
  return param->UpdateAllowUnknown(m);
}
}  // namespace xgboost
#endif  // XGBOOST_JSON_H_
