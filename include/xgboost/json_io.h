/**
 * Copyright 2019-2024, XGBoost Contributors
 */
#ifndef XGBOOST_JSON_IO_H_
#define XGBOOST_JSON_IO_H_
#include <dmlc/endian.h>
#include <xgboost/base.h>
#include <xgboost/json.h>

#include <cstdint>  // for int8_t
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace xgboost {
/**
 * \brief A json reader, currently error checking and utf-8 is not fully supported.
 */
class JsonReader {
 public:
  using Char = std::int8_t;

 protected:
  size_t constexpr static kMaxNumLength = std::numeric_limits<double>::max_digits10 + 1;

  struct SourceLocation {
   private:
    std::size_t pos_{0};  // current position in raw_str_

   public:
    SourceLocation() = default;
    size_t Pos() const { return pos_; }

    void Forward() { pos_++; }
    void Forward(uint32_t n) { pos_ += n; }
  } cursor_;

  StringView raw_str_;

 protected:
  void SkipSpaces();

  Char GetNextChar() {
    if (XGBOOST_EXPECT((cursor_.Pos() == raw_str_.size()), false)) {
      return -1;
    }
    char ch = raw_str_[cursor_.Pos()];
    cursor_.Forward();
    return ch;
  }

  Char PeekNextChar() {
    if (cursor_.Pos() == raw_str_.size()) {
      return -1;
    }
    Char ch = raw_str_[cursor_.Pos()];
    return ch;
  }

  /* \brief Skip spaces and consume next character. */
  Char GetNextNonSpaceChar() {
    SkipSpaces();
    return GetNextChar();
  }
  /* \brief Consume next character without first skipping empty space, throw when the next
   *        character is not the expected one.
   */
  Char GetConsecutiveChar(char expected_char) {
    Char result = GetNextChar();
    if (XGBOOST_EXPECT(result != expected_char, false)) { Expect(expected_char, result); }
    return result;
  }

  void Error(std::string msg) const;

  // Report expected character
  void Expect(Char c, Char got) {
    std::string msg = "Expecting: \"";
    msg += c;
    msg += "\", got: \"";
    if (got == EOF) {
      msg += "EOF\"";
    } else if (got == 0) {
      msg += "\\0\"";
    } else {
      msg += std::to_string(got) + " \"";
    }
    Error(msg);
  }

  virtual Json ParseString();
  virtual Json ParseObject();
  virtual Json ParseArray();
  virtual Json ParseNumber();
  virtual Json ParseBoolean();
  virtual Json ParseNull();

  Json Parse();

 public:
  explicit JsonReader(StringView str) :
      raw_str_{str} {}

  virtual ~JsonReader() = default;

  virtual Json Load();
};

class JsonWriter {
  template <typename T, std::enable_if_t<!std::is_same_v<Json, T>>* = nullptr>
  void Save(T const& v) {
    this->Save(Json{v});
  }
  template <typename Array, typename Fn>
  void WriteArray(Array const* arr, Fn&& fn) {
    stream_->emplace_back('[');
    auto const& vec = arr->GetArray();
    size_t size = vec.size();
    for (size_t i = 0; i < size; ++i) {
      auto const& value = vec[i];
      this->Save(fn(value));
      if (i != size - 1) {
        stream_->emplace_back(',');
      }
    }
    stream_->emplace_back(']');
  }

 protected:
  std::vector<char>* stream_;

 public:
  explicit JsonWriter(std::vector<char>* stream) : stream_{stream} {}

  virtual ~JsonWriter() = default;

  virtual void Save(Json json);

  virtual void Visit(JsonArray  const* arr);
  virtual void Visit(F32Array  const* arr);
  virtual void Visit(F64Array const*) { LOG(FATAL) << "Only UBJSON format can handle f64 array."; }
  virtual void Visit(I8Array  const* arr);
  virtual void Visit(U8Array const* arr);
  virtual void Visit(I16Array const* arr);
  virtual void Visit(U16Array const* arr);
  virtual void Visit(I32Array  const* arr);
  virtual void Visit(U32Array  const* arr);
  virtual void Visit(I64Array  const* arr);
  virtual void Visit(U64Array  const* arr);
  virtual void Visit(JsonObject const* obj);
  virtual void Visit(JsonNumber const* num);
  virtual void Visit(JsonInteger const* num);
  virtual void Visit(JsonNull   const* null);
  virtual void Visit(JsonString const* str);
  virtual void Visit(JsonBoolean const* boolean);
};

#if defined(__GLIBC__)
template <typename T>
T BuiltinBSwap(T v);

template <>
inline uint16_t BuiltinBSwap(uint16_t v) {
  return __builtin_bswap16(v);
}

template <>
inline uint32_t BuiltinBSwap(uint32_t v) {
  return __builtin_bswap32(v);
}

template <>
inline uint64_t BuiltinBSwap(uint64_t v) {
  return __builtin_bswap64(v);
}
#else
template <typename T>
T BuiltinBSwap(T v) {
  dmlc::ByteSwap(&v, sizeof(v), 1);
  return v;
}
#endif  //  defined(__GLIBC__)

template <typename T, std::enable_if_t<sizeof(T) == 1>* = nullptr>
inline T ToBigEndian(T v) {
  return v;
}

template <typename T, std::enable_if_t<sizeof(T) != 1>* = nullptr>
inline T ToBigEndian(T v) {
  static_assert(std::is_pod<T>::value, "Only pod is supported.");
#if DMLC_LITTLE_ENDIAN
  auto constexpr kS = sizeof(T);
  std::conditional_t<kS == 2, uint16_t, std::conditional_t<kS == 4, uint32_t, uint64_t>> u;
  std::memcpy(&u, &v, sizeof(u));
  u = BuiltinBSwap(u);
  std::memcpy(&v, &u, sizeof(u));
#endif  // DMLC_LITTLE_ENDIAN
  return v;
}

/**
 * \brief Reader for UBJSON https://ubjson.org/
 */
class UBJReader : public JsonReader {
  Json Parse();

  template <typename T>
  T ReadStream() {
    auto ptr = this->raw_str_.c_str() + cursor_.Pos();
    T v{0};
    std::memcpy(&v, ptr, sizeof(v));
    cursor_.Forward(sizeof(v));
    return v;
  }

  template <typename T>
  T ReadPrimitive() {
    auto v = ReadStream<T>();
    v = ToBigEndian(v);
    return v;
  }

  template <typename TypedArray>
  auto ParseTypedArray(std::int64_t n) {
    TypedArray results{static_cast<size_t>(n)};
    for (int64_t i = 0; i < n; ++i) {
      auto v = this->ReadPrimitive<typename TypedArray::value_type>();
      results.Set(i, v);
    }
    return Json{std::move(results)};
  }

  std::string DecodeStr();

  Json ParseArray() override;
  Json ParseObject() override;

 public:
  using JsonReader::JsonReader;
  Json Load() override;
};

/**
 * \brief Writer for UBJSON https://ubjson.org/
 */
class UBJWriter : public JsonWriter {
  void Visit(JsonArray const* arr) override;
  void Visit(F32Array const* arr) override;
  void Visit(F64Array const* arr) override;
  void Visit(I8Array  const* arr) override;
  void Visit(U8Array  const* arr) override;
  void Visit(I16Array  const* arr) override;
  void Visit(I32Array  const* arr) override;
  void Visit(I64Array  const* arr) override;
  void Visit(JsonObject const* obj) override;
  void Visit(JsonNumber const* num) override;
  void Visit(JsonInteger const* num) override;
  void Visit(JsonNull const* null) override;
  void Visit(JsonString const* str) override;
  void Visit(JsonBoolean const* boolean) override;

 public:
  using JsonWriter::JsonWriter;
  void Save(Json json) override;
};
}      // namespace xgboost

#endif  // XGBOOST_JSON_IO_H_
