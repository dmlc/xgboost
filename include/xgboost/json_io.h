/*!
 * Copyright (c) by Contributors 2019
 */
#ifndef XGBOOST_JSON_IO_H_
#define XGBOOST_JSON_IO_H_
#include <xgboost/json.h>
#include <xgboost/base.h>

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <map>
#include <limits>
#include <sstream>
#include <locale>
#include <cinttypes>

namespace xgboost {
/*
 * \brief A json reader, currently error checking and utf-8 is not fully supported.
 */
class JsonReader {
 protected:
  size_t constexpr static kMaxNumLength =
      std::numeric_limits<double>::max_digits10 + 1;

  struct SourceLocation {
   private:
    size_t pos_ { 0 };  // current position in raw_str_

   public:
    SourceLocation() = default;
    size_t  Pos()  const { return pos_; }

    void Forward() {
      pos_++;
    }
    void Forward(uint32_t n) {
      pos_ += n;
    }
  } cursor_;

  StringView raw_str_;

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

  /* \brief Skip spaces and consume next character. */
  char GetNextNonSpaceChar() {
    SkipSpaces();
    return GetNextChar();
  }
  /* \brief Consume next character without first skipping empty space, throw when the next
   *        character is not the expected one.
   */
  char GetConsecutiveChar(char expected_char) {
    char result = GetNextChar();
    if (XGBOOST_EXPECT(result != expected_char, false)) { Expect(expected_char, result); }
    return result;
  }

  void Error(std::string msg) const;

  // Report expected character
  void Expect(char c, char got) {
    std::string msg = "Expecting: \"";
    msg += c;
    msg += "\", got: \"";
    if (got == -1) {
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

  Json Load();
};

class JsonWriter {
  static constexpr size_t kIndentSize = 2;

  size_t n_spaces_;
  std::vector<char>* stream_;

 public:
  explicit JsonWriter(std::vector<char>* stream) :
      n_spaces_{0}, stream_{stream} {}

  virtual ~JsonWriter() = default;

  void Save(Json json);

  virtual void Visit(JsonArray  const* arr);
  virtual void Visit(JsonObject const* obj);
  virtual void Visit(JsonNumber const* num);
  virtual void Visit(JsonInteger const* num);
  virtual void Visit(JsonNull   const* null);
  virtual void Visit(JsonString const* str);
  virtual void Visit(JsonBoolean const* boolean);
};
}      // namespace xgboost

#endif  // XGBOOST_JSON_IO_H_
