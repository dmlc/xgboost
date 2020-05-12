/*!
 * Copyright (c) by Contributors 2019
 */
#include <cctype>
#include <locale>
#include <sstream>
#include <limits>
#include <cmath>

#include "xgboost/base.h"
#include "xgboost/logging.h"
#include "xgboost/json.h"
#include "xgboost/json_io.h"

namespace xgboost {

void JsonWriter::Save(Json json) {
  json.ptr_->Save(this);
}

void JsonWriter::Visit(JsonArray const* arr) {
  this->Write("[");
  auto const& vec = arr->GetArray();
  size_t size = vec.size();
  for (size_t i = 0; i < size; ++i) {
    auto const& value = vec[i];
    this->Save(value);
    if (i != size-1) { Write(","); }
  }
  this->Write("]");
}

void JsonWriter::Visit(JsonObject const* obj) {
  this->Write("{");
  this->BeginIndent();
  this->NewLine();

  size_t i = 0;
  size_t size = obj->GetObject().size();

  for (auto& value : obj->GetObject()) {
    this->Write("\"" + value.first + "\":");
    this->Save(value.second);

    if (i != size-1) {
      this->Write(",");
      this->NewLine();
    }
    i++;
  }
  this->EndIndent();
  this->NewLine();
  this->Write("}");
}

void JsonWriter::Visit(JsonNumber const* num) {
  convertor_ << num->GetNumber();
  auto const& str = convertor_.str();
  this->Write(StringView{str.c_str(), str.size()});
  convertor_.str("");
}

void JsonWriter::Visit(JsonInteger const* num) {
  convertor_ << num->GetInteger();
  auto const& str = convertor_.str();
  this->Write(StringView{str.c_str(), str.size()});
  convertor_.str("");
}

void JsonWriter::Visit(JsonNull const* null) {
  this->Write("null");
}

void JsonWriter::Visit(JsonString const* str) {
  std::string buffer;
  buffer += '"';
  auto const& string = str->GetString();
  for (size_t i = 0; i < string.length(); i++) {
    const char ch = string[i];
    if (ch == '\\') {
      if (i < string.size() && string[i+1] == 'u') {
        buffer += "\\";
      } else {
        buffer += "\\\\";
      }
    } else if (ch == '"') {
      buffer += "\\\"";
    } else if (ch == '\b') {
      buffer += "\\b";
    } else if (ch == '\f') {
      buffer += "\\f";
    } else if (ch == '\n') {
      buffer += "\\n";
    } else if (ch == '\r') {
      buffer += "\\r";
    } else if (ch == '\t') {
      buffer += "\\t";
    } else if (static_cast<uint8_t>(ch) <= 0x1f) {
      // Unit separator
      char buf[8];
      snprintf(buf, sizeof buf, "\\u%04x", ch);
      buffer += buf;
    } else {
      buffer += ch;
    }
  }
  buffer += '"';
  this->Write(buffer);
}

void JsonWriter::Visit(JsonBoolean const* boolean) {
  bool val = boolean->GetBoolean();
  if (val) {
    this->Write(u8"true");
  } else {
    this->Write(u8"false");
  }
}

// Value
std::string Value::TypeStr() const {
  switch (kind_) {
    case ValueKind::kString:  return "String";  break;
    case ValueKind::kNumber:  return "Number";  break;
    case ValueKind::kObject:  return "Object";  break;
    case ValueKind::kArray:   return "Array";   break;
    case ValueKind::kBoolean: return "Boolean"; break;
    case ValueKind::kNull:    return "Null";    break;
    case ValueKind::kInteger: return "Integer"; break;
  }
  return "";
}

// Only used for keeping old compilers happy about non-reaching return
// statement.
Json& DummyJsonObject() {
  static Json obj;
  return obj;
}

// Json Object
JsonObject::JsonObject(JsonObject && that) :
    Value(ValueKind::kObject), object_{std::move(that.object_)} {}

JsonObject::JsonObject(std::map<std::string, Json>&& object)
    : Value(ValueKind::kObject), object_{std::move(object)} {}

Json& JsonObject::operator[](std::string const & key) {
  return object_[key];
}

Json& JsonObject::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonObject::operator==(Value const& rhs) const {
  if (!IsA<JsonObject>(&rhs)) { return false; }
  return object_ == Cast<JsonObject const>(&rhs)->GetObject();
}

Value& JsonObject::operator=(Value const &rhs) {
  JsonObject const* casted = Cast<JsonObject const>(&rhs);
  object_ = casted->GetObject();
  return *this;
}

void JsonObject::Save(JsonWriter* writer) {
  writer->Visit(this);
}

// Json String
Json& JsonString::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonString::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer."
             << "  Please try obtaining std::string first.";
  return DummyJsonObject();
}

bool JsonString::operator==(Value const& rhs) const {
  if (!IsA<JsonString>(&rhs)) { return false; }
  return Cast<JsonString const>(&rhs)->GetString() == str_;
}

Value & JsonString::operator=(Value const &rhs) {
  JsonString const* casted = Cast<JsonString const>(&rhs);
  str_ = casted->GetString();
  return *this;
}

// FIXME: UTF-8 parsing support.
void JsonString::Save(JsonWriter* writer) {
  writer->Visit(this);
}

// Json Array
JsonArray::JsonArray(JsonArray && that) :
    Value(ValueKind::kArray), vec_{std::move(that.vec_)} {}

Json& JsonArray::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonArray::operator[](int ind) {
  return vec_.at(ind);
}

bool JsonArray::operator==(Value const& rhs) const {
  if (!IsA<JsonArray>(&rhs)) { return false; }
  auto& arr = Cast<JsonArray const>(&rhs)->GetArray();
  return std::equal(arr.cbegin(), arr.cend(), vec_.cbegin());
}

Value & JsonArray::operator=(Value const &rhs) {
  JsonArray const* casted = Cast<JsonArray const>(&rhs);
  vec_ = casted->GetArray();
  return *this;
}

void JsonArray::Save(JsonWriter* writer) {
  writer->Visit(this);
}

// Json Number
Json& JsonNumber::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonNumber::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonNumber::operator==(Value const& rhs) const {
  if (!IsA<JsonNumber>(&rhs)) { return false; }
  return std::abs(number_ - Cast<JsonNumber const>(&rhs)->GetNumber()) < kRtEps;
}

Value & JsonNumber::operator=(Value const &rhs) {
  JsonNumber const* casted = Cast<JsonNumber const>(&rhs);
  number_ = casted->GetNumber();
  return *this;
}

void JsonNumber::Save(JsonWriter* writer) {
  writer->Visit(this);
}

// Json Integer
Json& JsonInteger::operator[](std::string const& key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonInteger::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonInteger::operator==(Value const& rhs) const {
  if (!IsA<JsonInteger>(&rhs)) { return false; }
  return integer_ == Cast<JsonInteger const>(&rhs)->GetInteger();
}

Value & JsonInteger::operator=(Value const &rhs) {
  JsonInteger const* casted = Cast<JsonInteger const>(&rhs);
  integer_ = casted->GetInteger();
  return *this;
}

void JsonInteger::Save(JsonWriter* writer) {
  writer->Visit(this);
}

// Json Null
Json& JsonNull::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonNull::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonNull::operator==(Value const& rhs) const {
  if (!IsA<JsonNull>(&rhs)) { return false; }
  return true;
}

Value & JsonNull::operator=(Value const &rhs) {
  Cast<JsonNull const>(&rhs);  // Checking only.
  return *this;
}

void JsonNull::Save(JsonWriter* writer) {
  writer->Write("null");
}

// Json Boolean
Json& JsonBoolean::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonBoolean::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonBoolean::operator==(Value const& rhs) const {
  if (!IsA<JsonBoolean>(&rhs)) { return false; }
  return boolean_ == Cast<JsonBoolean const>(&rhs)->GetBoolean();
}

Value & JsonBoolean::operator=(Value const &rhs) {
  JsonBoolean const* casted = Cast<JsonBoolean const>(&rhs);
  boolean_ = casted->GetBoolean();
  return *this;
}

void JsonBoolean::Save(JsonWriter *writer) {
  writer->Visit(this);
}

size_t constexpr JsonReader::kMaxNumLength;

Json JsonReader::Parse() {
  while (true) {
    SkipSpaces();
    char c = PeekNextChar();
    if (c == -1) { break; }

    if (c == '{') {
      return ParseObject();
    } else if ( c == '[' ) {
      return ParseArray();
    } else if ( c == '-' || std::isdigit(c) ||
                c == 'N' ) {
      // For now we only accept `NaN`, not `nan` as the later violiates LR(1) with `null`.
      return ParseNumber();
    } else if ( c == '\"' ) {
      return ParseString();
    } else if ( c == 't' || c == 'f' ) {
      return ParseBoolean();
    } else if (c == 'n') {
      return ParseNull();
    } else {
      Error("Unknown construct");
    }
  }
  return Json();
}

Json JsonReader::Load() {
  Json result = Parse();
  return result;
}

void JsonReader::Error(std::string msg) const {
  // just copy it.
  std::istringstream str_s(raw_str_.substr(0, raw_str_.size()));

  msg += ", around character: " + std::to_string(cursor_.Pos());
  msg += '\n';

  constexpr size_t kExtend = 8;
  auto beg = static_cast<int64_t>(cursor_.Pos()) -
             static_cast<int64_t>(kExtend) < 0 ? 0 : cursor_.Pos() - kExtend;
  auto end = cursor_.Pos() + kExtend >= raw_str_.size() ?
             raw_str_.size() : cursor_.Pos() + kExtend;

  std::string const& raw_portion = raw_str_.substr(beg, end - beg);
  std::string portion;
  for (auto c : raw_portion) {
    if (c == '\n') {
      portion += "\\n";
    } else {
      portion += c;
    }
  }

  msg += "    ";
  msg += portion;
  msg += '\n';

  msg += "    ";
  for (size_t i = beg; i < cursor_.Pos() - 1; ++i) {
    msg += '~';
  }
  msg += '^';
  for (size_t i = cursor_.Pos(); i < end; ++i) {
    msg += '~';
  }
  LOG(FATAL) << msg;
}

// Json class
void JsonReader::SkipSpaces() {
  while (cursor_.Pos() < raw_str_.size()) {
    char c = raw_str_[cursor_.Pos()];
    if (std::isspace(c)) {
      cursor_.Forward();
    } else {
      break;
    }
  }
}

void ParseStr(std::string const& str) {
  size_t end = 0;
  for (size_t i = 0; i < str.size(); ++i) {
    if (str[i] == '"' && i > 0 && str[i-1] != '\\') {
      end = i;
      break;
    }
  }
  std::string result;
  result.resize(end);
}

Json JsonReader::ParseString() {
  char ch { GetChar('\"') };  // NOLINT
  std::ostringstream output;
  std::string str;
  while (true) {
    ch = GetNextChar();
    if (ch == '\\') {
      char next = static_cast<char>(GetNextChar());
      switch (next) {
        case 'r':  str += u8"\r"; break;
        case 'n':  str += u8"\n"; break;
        case '\\': str += u8"\\"; break;
        case 't':  str += u8"\t"; break;
        case '\"': str += u8"\""; break;
        case 'u':
          str += ch;
          str += 'u';
          break;
        default: Error("Unknown escape");
      }
    } else {
      if (ch == '\"') break;
      str += ch;
    }
    if (ch == EOF || ch == '\r' || ch == '\n') {
      Expect('\"', ch);
    }
  }
  return Json(std::move(str));
}

Json JsonReader::ParseNull() {
  char ch = GetNextNonSpaceChar();
  std::string buffer{ch};
  for (size_t i = 0; i < 3; ++i) {
    buffer.push_back(GetNextChar());
  }
  if (buffer != "null") {
    Error("Expecting null value \"null\"");
  }
  return Json{JsonNull()};
}

Json JsonReader::ParseArray() {
  std::vector<Json> data;

  char ch { GetChar('[') };  // NOLINT
  while (true) {
    if (PeekNextChar() == ']') {
      GetChar(']');
      return Json(std::move(data));
    }
    auto obj = Parse();
    data.push_back(obj);
    ch = GetNextNonSpaceChar();
    if (ch == ']') break;
    if (ch != ',') {
      Expect(',', ch);
    }
  }

  return Json(std::move(data));
}

Json JsonReader::ParseObject() {
  GetChar('{');

  std::map<std::string, Json> data;
  SkipSpaces();
  char ch = PeekNextChar();

  if (ch == '}') {
    GetChar('}');
    return Json(std::move(data));
  }

  while (true) {
    SkipSpaces();
    ch = PeekNextChar();
    CHECK_NE(ch, -1) << "cursor_.Pos(): " << cursor_.Pos() << ", "
                     << "raw_str_.size():" << raw_str_.size();
    if (ch != '"') {
      Expect('"', ch);
    }
    Json key = ParseString();

    ch = GetNextNonSpaceChar();

    if (ch != ':') {
      Expect(':', ch);
    }

    Json value { Parse() };

    data[get<String>(key)] = std::move(value);

    ch = GetNextNonSpaceChar();

    if (ch == '}') break;
    if (ch != ',') {
      Expect(',', ch);
    }
  }

  return Json(std::move(data));
}

Json JsonReader::ParseNumber() {
  // Adopted from sajson with some simplifications and small optimizations.
  char const* p = raw_str_.c_str() + cursor_.Pos();
  char const* const beg = p;  // keep track of current pointer

  // TODO(trivialfis): Add back all the checks for number
  bool negative = false;
  if (XGBOOST_EXPECT(*p == 'N', false)) {
    GetChar('N');
    GetChar('a');
    GetChar('N');
    return Json(static_cast<Number::Float>(std::numeric_limits<float>::quiet_NaN()));
  }

  if ('-' == *p) {
    ++p;
    negative = true;
  }

  bool is_float = false;

  using ExpInt = std::remove_const<
    decltype(std::numeric_limits<Number::Float>::max_exponent)>::type;
  constexpr auto kExpMax = std::numeric_limits<ExpInt>::max();
  constexpr auto kExpMin = std::numeric_limits<ExpInt>::min();

  JsonInteger::Int i = 0;
  double f = 0.0;  // Use double to maintain accuracy

  if (*p == '0') {
    ++p;
  } else {
    char c = *p;
    do {
      ++p;
      char digit = c - '0';
      i = 10 * i + digit;
      c = *p;
    } while (std::isdigit(c));
  }

  ExpInt exponent = 0;
  const char *const dot_position = p;
  if ('.' == *p) {
    is_float = true;
    f = i;
    ++p;
    char c = *p;

    do {
      ++p;
      f = f * 10 + (c - '0');
      c = *p;
    } while (std::isdigit(c));
  }
  if (is_float) {
    exponent = dot_position - p + 1;
  }

  char e = *p;
  if ('e' == e || 'E' == e) {
    if (!is_float) {
      is_float = true;
      f = i;
    }
    ++p;

    bool negative_exponent = false;
    if ('-' == *p) {
      negative_exponent = true;
      ++p;
    } else if ('+' == *p) {
      ++p;
    }

    ExpInt exp = 0;

    char c = *p;
    while (std::isdigit(c)) {
      unsigned char digit = c - '0';
      if (XGBOOST_EXPECT(exp > (kExpMax - digit) / 10, false)) {
        CHECK_GT(exp, (kExpMax - digit) / 10) << "Overflow";
      }
      exp = 10 * exp + digit;
      ++p;
      c = *p;
    }
    static_assert(-kExpMax >= kExpMin, "exp can be negated without loss or UB");
    exponent += (negative_exponent ? -exp : exp);
  }

  if (exponent) {
    CHECK(is_float);
    // If d is zero but the exponent is huge, don't
    // multiply zero by inf which gives nan.
    if (f != 0.0) {
      // Only use exp10 from libc on gcc+linux
#if !defined(__GNUC__) || defined(_WIN32) || defined(__APPLE__) || !defined(__linux__)
#define exp10(val) std::pow(10, (val))
#endif  // !defined(__GNUC__) || defined(_WIN32) || defined(__APPLE__) || !defined(__linux__)
      f *= exp10(exponent);
#if !defined(__GNUC__) || defined(_WIN32) || defined(__APPLE__) || !defined(__linux__)
#undef exp10
#endif  // !defined(__GNUC__) || defined(_WIN32) || defined(__APPLE__) || !defined(__linux__)
    }
  }

  if (negative) {
      f = -f;
      i = -i;
  }

  auto moved = std::distance(beg, p);
  this->cursor_.Forward(moved);

  if (is_float) {
    return Json(static_cast<Number::Float>(f));
  } else {
    return Json(JsonInteger(i));
  }
}

Json JsonReader::ParseBoolean() {
  bool result = false;
  char ch = GetNextNonSpaceChar();
  std::string const t_value = u8"true";
  std::string const f_value = u8"false";
  std::string buffer;

  if (ch == 't') {
    for (size_t i = 0; i < 3; ++i) {
      buffer.push_back(GetNextNonSpaceChar());
    }
    if (buffer != u8"rue") {
      Error("Expecting boolean value \"true\".");
    }
    result = true;
  } else {
    for (size_t i = 0; i < 4; ++i) {
      buffer.push_back(GetNextNonSpaceChar());
    }
    if (buffer != u8"alse") {
      Error("Expecting boolean value \"false\".");
    }
    result = false;
  }
  return Json{JsonBoolean{result}};
}

Json Json::Load(StringView str) {
  JsonReader reader(str);
  Json json{reader.Load()};
  return json;
}

Json Json::Load(JsonReader* reader) {
  Json json{reader->Load()};
  return json;
}

void Json::Dump(Json json, std::ostream *stream, bool pretty) {
  JsonWriter writer(stream, pretty);
  writer.Save(json);
}

void Json::Dump(Json json, std::string* str, bool pretty) {
  std::stringstream ss;
  JsonWriter writer(&ss, pretty);
  writer.Save(json);
  *str = ss.str();
}

Json& Json::operator=(Json const &other) = default;
}  // namespace xgboost
