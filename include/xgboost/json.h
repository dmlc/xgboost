/*!
 * Copyright (c) by XGBoost Contributors 2019-2021
 */
#ifndef XGBOOST_JSON_H_
#define XGBOOST_JSON_H_

#include <xgboost/logging.h>
#include <xgboost/parameter.h>
#include <xgboost/intrusive_ptr.h>

#include <map>
#include <memory>
#include <vector>
#include <functional>
#include <utility>
#include <string>

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
    kNull
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
  return T::IsClassOf(value);
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
  JsonString() : Value(ValueKind::kString) {}
  JsonString(std::string const& str) :  // NOLINT
      Value(ValueKind::kString), str_{str} {}
  JsonString(std::string&& str) :  // NOLINT
      Value(ValueKind::kString), str_{std::move(str)} {}
  JsonString(JsonString&& str) noexcept :  // NOLINT
      Value(ValueKind::kString), str_{std::move(str.str_)} {}

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  std::string const& GetString() &&      { return str_; }
  std::string const& GetString() const & { return str_; }
  std::string&       GetString()       & { return str_; }

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kString;
  }
};

class JsonArray : public Value {
  std::vector<Json> vec_;

 public:
  JsonArray() : Value(ValueKind::kArray) {}
  JsonArray(std::vector<Json>&& arr) :  // NOLINT
      Value(ValueKind::kArray), vec_{std::move(arr)} {}
  JsonArray(std::vector<Json> const& arr) :  // NOLINT
      Value(ValueKind::kArray), vec_{arr} {}
  JsonArray(JsonArray const& that) = delete;
  JsonArray(JsonArray && that);

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  std::vector<Json> const& GetArray() &&      { return vec_; }
  std::vector<Json> const& GetArray() const & { return vec_; }
  std::vector<Json>&       GetArray()       & { return vec_; }

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kArray;
  }
};

class JsonObject : public Value {
  std::map<std::string, Json> object_;

 public:
  JsonObject() : Value(ValueKind::kObject) {}
  JsonObject(std::map<std::string, Json>&& object);  // NOLINT
  JsonObject(JsonObject const& that) = delete;
  JsonObject(JsonObject && that);

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  std::map<std::string, Json> const& GetObject() &&      { return object_; }
  std::map<std::string, Json> const& GetObject() const & { return object_; }
  std::map<std::string, Json> &      GetObject() &       { return object_; }

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kObject;
  }
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

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  Float const& GetNumber() &&      { return number_; }
  Float const& GetNumber() const & { return number_; }
  Float&       GetNumber()       & { return number_; }


  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

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

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  Int const& GetInteger() &&      { return integer_; }
  Int const& GetInteger() const & { return integer_; }
  Int& GetInteger() &             { return integer_; }
  void Save(JsonWriter* writer) override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kInteger;
  }
};

class JsonNull : public Value {
 public:
  JsonNull() : Value(ValueKind::kNull) {}
  JsonNull(std::nullptr_t) : Value(ValueKind::kNull) {}  // NOLINT
  JsonNull(JsonNull&&) noexcept : Value(ValueKind::kNull) {}

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

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

  void Save(JsonWriter* writer) override;

  Json& operator[](std::string const & key) override;
  Json& operator[](int ind) override;

  bool const& GetBoolean() &&      { return boolean_; }
  bool const& GetBoolean() const & { return boolean_; }
  bool&       GetBoolean()       & { return boolean_; }

  bool operator==(Value const& rhs) const override;
  Value& operator=(Value const& rhs) override;

  static bool IsClassOf(Value const* value) {
    return value->Type() == ValueKind::kBoolean;
  }
};

struct StringView {
 private:
  using CharT = char;  // unsigned char
  using Traits = std::char_traits<CharT>;
  CharT const* str_;
  size_t size_;

 public:
  StringView() = default;
  StringView(CharT const* str, size_t size) : str_{str}, size_{size} {}
  explicit StringView(std::string const& str): str_{str.c_str()}, size_{str.size()} {}
  explicit StringView(CharT const* str) : str_{str}, size_{Traits::length(str)} {}

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
  CharT const* c_str() const { return str_; }  // NOLINT

  CharT const* cbegin() const { return str_; }         // NOLINT
  CharT const* cend() const { return str_ + size(); }  // NOLINT
  CharT const* begin() const { return str_; }          // NOLINT
  CharT const* end() const { return str_ + size(); }   // NOLINT
};

std::ostream &operator<<(std::ostream &os, StringView const v);

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
  static void Dump(Json json, std::string* out);

  Json() : ptr_{new JsonNull} {}

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
  Json(Json const& other) = default;
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

 private:
  IntrusivePtr<Value> ptr_;
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

// Object
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonObject>::value>::type* = nullptr>
std::map<std::string, Json>& GetImpl(T& val) {  // NOLINT
  return val.GetObject();
}
template <typename T,
          typename std::enable_if<
            std::is_same<T, JsonObject const>::value>::type* = nullptr>
std::map<std::string, Json> const& GetImpl(T& val) {  // NOLINT
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
template <typename Parameter>
Object ToJson(Parameter const& param) {
  Object obj;
  for (auto const& kv : param.__DICT__()) {
    obj[kv.first] = kv.second;
  }
  return obj;
}

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
