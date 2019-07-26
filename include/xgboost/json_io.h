/*!
 * Copyright (c) by Contributors 2019
 */
#ifndef XGBOOST_JSON_IO_H_
#define XGBOOST_JSON_IO_H_
#include <xgboost/json.h>

#include <memory>
#include <string>
#include <cinttypes>
#include <utility>
#include <map>
#include <limits>
#include <sstream>
#include <locale>

namespace xgboost {

template <typename Allocator>
class FixedPrecisionStreamContainer : public std::basic_stringstream<
  char, std::char_traits<char>, Allocator> {
 public:
  FixedPrecisionStreamContainer() {
    this->precision(std::numeric_limits<Number::Float>::max_digits10);
  }
};

using FixedPrecisionStream = FixedPrecisionStreamContainer<std::allocator<char>>;

/*
 * \brief An reader that can be specialised.
 *
 * Why specialization?
 *
 *   First of all, we don't like specialization.  This is purely for performance concern.
 *   Distributed environment freqently serializes model so at some point this could be a
 *   bottle neck for training performance.  There are many other techniques for obtaining
 *   better performance, but all of them requires implementing thier own allocaltor(s),
 *   using simd instructions.  And few of them can provide a easy to modify structure
 *   since they assumes a fixed memory layout.
 *
 *   In XGBoost we provide specialized logic for parsing/writing tree models and linear
 *   models, where dense numeric values is presented, including weights, node ids etc.
 *
 * Plan for removing the specialization:
 *
 *   We plan to upstream this implementaion into DMLC as it matures.  For XGBoost, most of
 *   the time spent in load/dump is actually `sprintf`.
 *
 * To enable specialization, register a keyword that corresponds to
 * key in Json object.  For example in:
 *
 * \code
 * { "key": {...} }
 * \endcode
 *
 * To add special logic for parsing {...}, one can call:
 *
 * \code
 * JsonReader::registry("key", [](StringView str, size_t* pos){ ... return JsonRaw(...); });
 * \endcode
 *
 * Where str is a view of entire input string, while pos is a pointer to current position.
 * The function must return a raw object.  Later after obtaining a parsed object, say
 * `Json obj`, you can obtain * the raw object by calling `obj["key"]' then perform the
 * specialized parsing on it.
 *
 * See `LinearSelectRaw` and `LinearReader` in combination as an example.
 */
class JsonReader {
 protected:
  size_t constexpr static kMaxNumLength =
      std::numeric_limits<double>::max_digits10 + 1;

  struct SourceLocation {
    size_t pos_;  // current position in raw_str_

   public:
    SourceLocation() : pos_(0) {}
    explicit SourceLocation(size_t pos) : pos_{pos} {}
    size_t  Pos()  const { return pos_; }

    SourceLocation& Forward(char c = 0) {
      pos_++;
      return *this;
    }
  } cursor_;

  StringView raw_str_;
  bool ignore_specialization_;

 protected:
  void SkipSpaces();

  char GetNextChar() {
    if (cursor_.Pos() == raw_str_.size()) {
      return -1;
    }
    char ch = raw_str_[cursor_.Pos()];
    cursor_.Forward();
    return ch;
  }

  char PeekNextChar() {
    if (cursor_.Pos() == raw_str_.size()) {
      return -1;
    }
    char ch = raw_str_[cursor_.Pos()];
    return ch;
  }

  char GetNextNonSpaceChar() {
    SkipSpaces();
    return GetNextChar();
  }

  char GetChar(char c) {
    char result = GetNextNonSpaceChar();
    if (result != c) { Expect(c, result); }
    return result;
  }

  void Error(std::string msg) const;

  // Report expected character
  void Expect(char c, char got) {
    std::string msg = "Expecting: \"";
    msg += c;
    msg += "\", got: \"";
    msg += std::string {got} + " \"";
    Error(msg);
  }

  virtual Json ParseString();
  virtual Json ParseObject();
  virtual Json ParseArray();
  virtual Json ParseNumber();
  virtual Json ParseBoolean();
  virtual Json ParseNull();

  Json Parse();

 private:
  using Fn = std::function<Json (StringView, size_t*)>;

 public:
  explicit JsonReader(StringView str, bool ignore = false) :
      raw_str_{str},
      ignore_specialization_{ignore} {}
  explicit JsonReader(StringView str, size_t pos, bool ignore = false) :
      cursor_{pos},
      raw_str_{str},
      ignore_specialization_{ignore} {}

  virtual ~JsonReader() = default;

  Json Load();

  static std::map<std::string, Fn>& getRegistry() {
    static std::map<std::string, Fn> set;
    return set;
  }

  static std::map<std::string, Fn> const& registry(
      std::string const& key, Fn fn) {
    getRegistry()[key] = fn;
    return getRegistry();
  }
};

class JsonWriter {
  static constexpr size_t kIndentSize = 2;
  FixedPrecisionStream convertor_;

  size_t n_spaces_;
  std::ostream* stream_;
  bool pretty_;

 public:
  JsonWriter(std::ostream* stream, bool pretty) :
      n_spaces_{0}, stream_{stream}, pretty_{pretty} {}

  virtual ~JsonWriter() = default;

  void NewLine() {
    if (pretty_) {
      *stream_ << u8"\n" << std::string(n_spaces_, ' ');
    }
  }

  void BeginIndent() {
    n_spaces_ += kIndentSize;
  }
  void EndIndent() {
    n_spaces_ -= kIndentSize;
  }

  void Write(std::string str) {
    *stream_ << str;
  }
  void Write(StringView str) {
    stream_->write(str.c_str(), str.size());
  }

  void Save(Json json);

  virtual void Visit(JsonArray  const* arr);
  virtual void Visit(JsonObject const* obj);
  virtual void Visit(JsonNumber const* num);
  virtual void Visit(JsonRaw    const* raw);
  virtual void Visit(JsonNull   const* null);
  virtual void Visit(JsonString const* str);
  virtual void Visit(JsonBoolean const* boolean);
};
}      // namespace xgboost

#endif  // XGBOOST_JSON_IO_H_
