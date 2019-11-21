/*!
 * Copyright 2019 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_JSON_READER_EXPERIMENTAL_H_
#define XGBOOST_COMMON_JSON_READER_EXPERIMENTAL_H_

#include <iomanip>
#include <map>
#include <cmath>
#include <limits>
#include <utility>

#include "xgboost/span.h"
#include "xgboost/logging.h"

#include "charconv.h"
#include "luts.h"
#include "json_experimental.h"

namespace xgboost {
namespace experimental {

inline double FastPath(double significand, int exp) noexcept(true) {
  if (exp < -308) {
    return 0.0;
  } else if (exp >= 0) {
    return significand * Exp10Lut(exp);
  } else {
    return significand / Exp10Lut(-exp);
  }
}

constexpr double FastPathLimit() {
  return static_cast<double>((static_cast<uint64_t>(1) << 53) - 1);
}

#define JSON_PARSER_ASSERT_RETURN(__ret)                                       \
  {                                                                            \
    if (errno_ != jError::kSuccess) {                                          \
      return __ret;                                                            \
    }                                                                          \
  }

inline double Strtod(double significand, int exp, char *beg, char *end) {
  double result{std::numeric_limits<double>::quiet_NaN()};
  // The technique is picked up from rapidjson, they implemented a big integer
  // type for slow full precision, here we just use strtod for slow parsing.
  // Fast path:
  // http://www.exploringbinary.com/fast-path-decimal-to-floating-point-conversion/
  if (exp > 22 && exp < 22 + 16) {
    // Fast path cases in disguise
    significand *= Exp10Lut(exp - 22);
    exp = 22;
  }

  if (exp >= -22 && exp <= 22 && significand <= FastPathLimit()) {  // 2^53 - 1
    result = FastPath(significand, exp);
    return result;
  }
  result = std::strtod(beg, &end);
  return result;
}

class JsonRecursiveReader {
  using Cursor = StringRef::iterator;

  Json* handler_;
  StringRef input_;
  Cursor cursor_;
  jError errno_{jError::kSuccess};

  bool IsSpace(char c) noexcept(true) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
  }

  Cursor SkipWhitespaces(Cursor p) noexcept(true) {
    for (;;) {
      if (XGBOOST_EXPECT(p == input_.cend(), false)) {
        return 0;
      } else if (IsSpace(*p)) {
        ++p;
      } else {
        return p;
      }
    }
  }

  void HandleNull(Json* null) noexcept(true) {
    bool ret = true;
    ret &= this->Skip(&cursor_, 'n');
    ret &= this->Skip(&cursor_, 'u');
    ret &= this->Skip(&cursor_, 'l');
    ret &= this->Skip(&cursor_, 'l');
    if (XGBOOST_EXPECT(!ret, false)) {
      errno_ = jError::kInvalidNull;
    }
    null->SetNull();
  }
  void HandleTrue(Json* t) noexcept(true) {
    bool ret = true;
    ret &= this->Skip(&cursor_, 't');
    ret &= this->Skip(&cursor_, 'r');
    ret &= this->Skip(&cursor_, 'u');
    ret &= this->Skip(&cursor_, 'e');
    if (XGBOOST_EXPECT(!ret, false)) {
      errno_ = jError::kInvalidTrue;
    }
    t->SetTrue();
  }
  void HandleFalse(Json* f) noexcept(true) {
    bool ret = true;
    ret &= this->Skip(&cursor_, 'f');
    ret &= this->Skip(&cursor_, 'a');
    ret &= this->Skip(&cursor_, 'l');
    ret &= this->Skip(&cursor_, 's');
    ret &= this->Skip(&cursor_, 'e');
    if (XGBOOST_EXPECT(!ret, false)) {
      errno_ = jError::kInvalidFalse;
    }
    f->SetFalse();
  }

  /*\brief Guess whether parsed value is floating point or integer.  For value
   * produced by nih json this should always be correct as ryu produces `E` in
   * all floating points. */
  void HandleNumber(Json* number) noexcept(true) {
    Cursor const beg = cursor_;  // keep track of current pointer

    bool negative = false;
    switch (*cursor_) {
    case '-': {
      negative = true;
      ++cursor_;
      break;
    }
    case '+': {
      negative = false;
      ++cursor_;
      break;
    }
    default: {
      break;
    }
    }

    bool is_float = false;

    int64_t i = 0;
    double f = 0.0;

    if (*cursor_ == '0') {
      i = 0;
      cursor_++;
    }

    while (XGBOOST_EXPECT(*cursor_ >= '0' && *cursor_ <= '9', true)) {
      i = i * 10 + (*cursor_ - '0');
      cursor_++;
    }

    int exp_frac {0};  // fraction of exponent

    if (*cursor_ == '.') {
      cursor_++;
      is_float = true;

      while (*cursor_ >= '0' && *cursor_ <= '9') {
        if (i > FastPathLimit()) {
          break;
        } else {
          i = i * 10 + (*cursor_ - '0');
          exp_frac--;
        }
        cursor_++;
      }

      f = static_cast<double>(i);
    }

    int exp {0};

    if (*cursor_ == 'E' || *cursor_ == 'e') {
      is_float = true;
      bool negative_exp {false};
      cursor_++;

      switch (*cursor_) {
      case '-': {
        negative_exp = true;
        cursor_++;
        break;
      }
      case '+': {
        cursor_++;
        break;
      }
      default:
        break;
      }

      if (XGBOOST_EXPECT(*cursor_ >= '0' && *cursor_ <= '9', true)) {
        exp = *cursor_ - '0';
        cursor_++;
        while (*cursor_ >= '0' && *cursor_ <= '9') {
          exp = exp * 10 + static_cast<int>(*cursor_ - '0');
          cursor_++;
        }
      } else {
        errno_ = jError::kInvalidNumber;
        return;;
      }

      if (negative_exp) {
        exp = -exp;
      }
    }

    if (negative) {
      i = -i;
    }
    if (is_float) {
      f = Strtod(i, exp + exp_frac, beg, cursor_);
      number->SetFloat(static_cast<float>(f));
    } else {
      number->SetInteger(i);
    }
  }

  ConstStringRef HandleString() noexcept(true) {
    auto ret = this->Skip(&cursor_, '\"');
    if (XGBOOST_EXPECT(!ret, false)) {
      errno_ = jError::kInvalidString;
      return {0, 0};
    }

    auto const beg = cursor_;
    bool still_parsing = true;
    bool has_escape = false;

    while (still_parsing) {
      char ch = *cursor_;
      if (XGBOOST_EXPECT(ch >= 0 && ch < 0x20, false)) {
        errno_ = jError::kInvalidString;
        return {};
      }

      switch (ch) {
      case '"': {
        still_parsing = false;
        break;
      }
      case '\\': {
        has_escape = true;

        if (XGBOOST_EXPECT(cursor_ + 1 == input_.cbegin(), false)) {
          errno_ = jError::kInvalidString;
          return {};
        }

        *cursor_ = '\0';  // mark invalid
        cursor_++;
        ch = *cursor_;

        switch (ch) {
        case 'r': {
          *cursor_ = '\r'; break;
        }
        case 't': {
          *cursor_ = '\t'; break;
        }
        case 'n': {
          *cursor_ = '\n'; break;
        }
        case 'b': {
          *cursor_ = '\b'; break;
        }
        case 'f': {
          *cursor_ = '\f'; break;
        }
        case '\\': {
          *cursor_ = '\\'; break;
        }
        case '\"': {
          *cursor_ = '\"'; break;
        }
        case '/': {
          *cursor_ = '/'; break;
        }
        }
        cursor_++;
        break;
      }
      default: {
        cursor_++;
        break;
      }
      }

      if (XGBOOST_EXPECT(!still_parsing, false)) {
        break;
      }
    }

    auto end = cursor_;
    if (XGBOOST_EXPECT(has_escape, false)) {
      Cursor last = beg;
      for (auto it = beg; it != cursor_; ++it) {
        if (XGBOOST_EXPECT(*it == '\0', false)) {
          continue;
        }
        *last = *it;
        last++;
      }
      end = last;
    }

    if (XGBOOST_EXPECT(!this->Skip(&cursor_, '"'), false)) {
      errno_ = jError::kInvalidString;
      return ConstStringRef{};
    }

    auto length = std::distance(beg, end);
    return ConstStringRef {beg, static_cast<size_t>(length)};
  }

  bool Skip(Cursor* p_cursor, char c) noexcept(true) {
    auto cursor = *p_cursor;
    // clang-tidy somehow believes this is null pointer.  There's a test for empty string
    // so disabling the lint error.
    auto o = *cursor;  // NOLINT
    (*p_cursor)++;     // NOLINT
    return c == o;
  }

  void HandleArray(Json* value) noexcept(true) {
    if (XGBOOST_EXPECT(!this->Skip(&cursor_, '['), false)) {
      errno_ = jError::kInvalidArray;
      return;
    }
    value->SetArray();
    value->SizeHint(8);

    cursor_ = this->SkipWhitespaces(cursor_);

    char ch = *cursor_;
    if (ch == ']') {
      this->Skip(&cursor_, ']');
      return;
    }

    while (true) {
      cursor_ = this->SkipWhitespaces(cursor_);

      auto elem = value->CreateArrayElem();
      this->ParseImpl(&elem);
      JSON_PARSER_ASSERT_RETURN();

      cursor_ = this->SkipWhitespaces(cursor_);

      if (*cursor_ == ']') {
        this->Skip(&cursor_, ']');
        break;
      }

      if (XGBOOST_EXPECT(!this->Skip(&cursor_, ','), false)) {
        errno_ = jError::kInvalidArray;
        return;
      }
    }
  }

  void HandleObject(Json* object) noexcept(true) {
    this->Skip(&cursor_, '{');
    cursor_ = SkipWhitespaces(cursor_);
    char ch = *cursor_;
    object->SetObject();

    if (ch == '}') {
      this->Skip(&cursor_, '}');
      return;
    }

    while (true) {
      cursor_ = SkipWhitespaces(cursor_);
      ch = *cursor_;

      ConstStringRef key = HandleString();
      JSON_PARSER_ASSERT_RETURN();

      auto value = object->CreateMember(key);

      cursor_ = this->SkipWhitespaces(cursor_);
      if (XGBOOST_EXPECT(!this->Skip(&cursor_, ':'), false)) {
        errno_ = jError::kInvalidObject;
        return;
      }

      this->ParseImpl(&value);
      JSON_PARSER_ASSERT_RETURN();

      cursor_ = this->SkipWhitespaces(cursor_);

      ch = *cursor_;
      if (ch == '}') {
        break;
      }
      if (XGBOOST_EXPECT(!this->Skip(&cursor_, ','), false)) {
        errno_  = jError::kInvalidObject;;
        return;
      }
    }

    if (XGBOOST_EXPECT(!this->Skip(&cursor_, '}'), false)) {
      errno_  = jError::kInvalidObject;;
      return;
    }
    return;
  }

 public:
  void ParseImpl(Json *value) noexcept(true) {
    cursor_ = this->SkipWhitespaces(cursor_);
    if (cursor_ == input_.data() + input_.size()) {
      return;
    }
    char c = *cursor_;

    switch (c) {
    case '{': {
      HandleObject(value);
    } break;
    case '[': {
      HandleArray(value);
      break;
    }
    case '-':
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9': {
      HandleNumber(value);
      break;
     }
     case '\"': {
       ConstStringRef str = HandleString();
       value->SetString(str);
       break;
     }
     case 't': {
       HandleTrue(value);
       break;
     }
     case 'f': {
       HandleFalse(value);
       break;
     }
     case 'n': {
       HandleNull(value);
       break;
     }
     default:
       errno_ = jError::kUnknownConstruct;
       return;
     }

     return;
  }

  std::pair<jError, size_t> Parse() {
    this->ParseImpl(handler_);
    return {errno_, std::distance(input_.cbegin(), cursor_)};
  }

  JsonRecursiveReader(StringRef str, Json* handler)
      : handler_{handler}, input_{str}, cursor_{input_.begin()} {
    CHECK_NE(input_.size(), 0) << "Empty JSON string.";
    CHECK(str.data());
  }
};

#undef JSON_PARSER_ASSERT_RETURN

}      // namespace experimental
}      // namespace xgboost
#endif  // XGBOOST_COMMON_JSON_READER_EXPERIMENTAL_H_
