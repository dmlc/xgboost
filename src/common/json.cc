/*!
 * Copyright (c) by Contributors 2019-2020
 */
#include <cctype>
#include <cstddef>
#include <iterator>
#include <locale>
#include <sstream>
#include <limits>
#include <cmath>

#include "charconv.h"
#include "xgboost/base.h"
#include "xgboost/logging.h"
#include "xgboost/json.h"
#include "xgboost/json_io.h"

namespace xgboost {

void JsonWriter::Save(Json json) {
  json.ptr_->Save(this);
}

void JsonWriter::Visit(JsonArray const* arr) {
  stream_->emplace_back('[');
  auto const& vec = arr->GetArray();
  size_t size = vec.size();
  for (size_t i = 0; i < size; ++i) {
    auto const& value = vec[i];
    this->Save(value);
    if (i != size - 1) {
      stream_->emplace_back(',');
    }
  }
  stream_->emplace_back(']');
}

void JsonWriter::Visit(JsonObject const* obj) {
  stream_->emplace_back('{');
  size_t i = 0;
  size_t size = obj->GetObject().size();

  for (auto& value : obj->GetObject()) {
    auto s = String{value.first};
    this->Visit(&s);
    stream_->emplace_back(':');
    this->Save(value.second);

    if (i != size-1) {
      stream_->emplace_back(',');
    }
    i++;
  }

  stream_->emplace_back('}');
}

void JsonWriter::Visit(JsonNumber const* num) {
  char number[NumericLimits<float>::kToCharsSize];
  auto res = to_chars(number, number + sizeof(number), num->GetNumber());
  auto end = res.ptr;
  auto ori_size = stream_->size();
  stream_->resize(stream_->size() + end - number);
  std::memcpy(stream_->data() + ori_size, number, end - number);
}

void JsonWriter::Visit(JsonInteger const* num) {
  char i2s_buffer_[NumericLimits<int64_t>::kToCharsSize];
  auto i = num->GetInteger();
  auto ret = to_chars(i2s_buffer_, i2s_buffer_ + NumericLimits<int64_t>::kToCharsSize, i);
  auto end = ret.ptr;
  CHECK(ret.ec == std::errc());
  auto digits = std::distance(i2s_buffer_, end);
  auto ori_size = stream_->size();
  stream_->resize(ori_size + digits);
  std::memcpy(stream_->data() + ori_size, i2s_buffer_, digits);
}

void JsonWriter::Visit(JsonNull const* ) {
    auto s = stream_->size();
    stream_->resize(s + 4);
    auto& buf = (*stream_);
    buf[s + 0] = 'n';
    buf[s + 1] = 'u';
    buf[s + 2] = 'l';
    buf[s + 3] = 'l';
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

  auto s = stream_->size();
  stream_->resize(s + buffer.size());
  std::memcpy(stream_->data() + s, buffer.data(), buffer.size());
}

void JsonWriter::Visit(JsonBoolean const* boolean) {
  bool val = boolean->GetBoolean();
  auto s = stream_->size();
  if (val) {
    stream_->resize(s + 4);
    auto& buf = (*stream_);
    buf[s + 0] = 't';
    buf[s + 1] = 'r';
    buf[s + 2] = 'u';
    buf[s + 3] = 'e';
  } else {
    stream_->resize(s + 5);
    auto& buf = (*stream_);
    buf[s + 0] = 'f';
    buf[s + 1] = 'a';
    buf[s + 2] = 'l';
    buf[s + 3] = 's';
    buf[s + 4] = 'e';
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

Json& JsonObject::operator[](int ) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonObject::operator==(Value const& rhs) const {
  if (!IsA<JsonObject>(&rhs)) {
    return false;
  }
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
Json& JsonString::operator[](std::string const& ) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonString::operator[](int ) {
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

Json& JsonArray::operator[](std::string const& ) {
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
Json& JsonNumber::operator[](std::string const& ) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonNumber::operator[](int ) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonNumber::operator==(Value const& rhs) const {
  if (!IsA<JsonNumber>(&rhs)) { return false; }
  auto r_num = Cast<JsonNumber const>(&rhs)->GetNumber();
  if (std::isinf(number_)) {
    return std::isinf(r_num);
  }
  if (std::isnan(number_)) {
    return std::isnan(r_num);
  }
  return number_ - r_num == 0;
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
Json& JsonInteger::operator[](std::string const& ) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonInteger::operator[](int ) {
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
Json& JsonNull::operator[](std::string const& ) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonNull::operator[](int ) {
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
  writer->Visit(this);
}

// Json Boolean
Json& JsonBoolean::operator[](std::string const& ) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonBoolean::operator[](int ) {
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
                c == 'N' || c == 'I') {
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

  msg += ", around character position: " + std::to_string(cursor_.Pos());
  msg += '\n';

  if (cursor_.Pos() == 0) {
    LOG(FATAL) << msg << ", \"" << str_s.str() << " \"";
  }

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
    } else if (c == '\0') {
      portion += "\\0";
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

namespace {
bool IsSpace(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
}  // anonymous namespace

// Json class
void JsonReader::SkipSpaces() {
  while (cursor_.Pos() < raw_str_.size()) {
    char c = raw_str_[cursor_.Pos()];
    if (IsSpace(c)) {
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
  char ch { GetConsecutiveChar('\"') };  // NOLINT
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

  char ch { GetConsecutiveChar('[') };  // NOLINT
  while (true) {
    if (PeekNextChar() == ']') {
      GetConsecutiveChar(']');
      return Json(std::move(data));
    }
    auto obj = Parse();
    data.emplace_back(obj);
    ch = GetNextNonSpaceChar();
    if (ch == ']') break;
    if (ch != ',') {
      Expect(',', ch);
    }
  }

  return Json(std::move(data));
}

Json JsonReader::ParseObject() {
  GetConsecutiveChar('{');

  std::map<std::string, Json> data;
  SkipSpaces();
  char ch = PeekNextChar();

  if (ch == '}') {
    GetConsecutiveChar('}');
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
  if (XGBOOST_EXPECT(*p == 'N', false)) {
    GetConsecutiveChar('N');
    GetConsecutiveChar('a');
    GetConsecutiveChar('N');
    return Json(static_cast<Number::Float>(std::numeric_limits<float>::quiet_NaN()));
  }

  bool negative = false;
  switch (*p) {
  case '-': {
    negative = true;
    ++p;
    break;
  }
  case '+': {
    negative = false;
    ++p;
    break;
  }
  default: {
    break;
  }
  }

  if (XGBOOST_EXPECT(*p == 'I', false)) {
    cursor_.Forward(std::distance(beg, p));  // +/-
    for (auto i : {'I', 'n', 'f', 'i', 'n', 'i', 't', 'y'}) {
      GetConsecutiveChar(i);
    }
    auto f = std::numeric_limits<float>::infinity();
    if (negative) {
      f = -f;
    }
    return Json(static_cast<Number::Float>(f));
  }

  bool is_float = false;

  int64_t i = 0;

  if (*p == '0') {
    i = 0;
    p++;
  }

  while (XGBOOST_EXPECT(*p >= '0' && *p <= '9', true)) {
    i = i * 10 + (*p - '0');
    p++;
  }

  if (*p == '.') {
    p++;
    is_float = true;

    while (*p >= '0' && *p <= '9') {
      i = i * 10 + (*p - '0');
      p++;
    }
  }

  if (*p == 'E' || *p == 'e') {
    is_float = true;
    p++;

    switch (*p) {
    case '-':
    case '+': {
      p++;
      break;
    }
    default:
      break;
    }

    if (XGBOOST_EXPECT(*p >= '0' && *p <= '9', true)) {
      p++;
      while (*p >= '0' && *p <= '9') {
        p++;
      }
    } else {
      Error("Expecting digit");
    }
  }

  auto moved = std::distance(beg, p);
  this->cursor_.Forward(moved);

  if (is_float) {
    float f;
    auto ret = from_chars(beg, p, f);
    if (XGBOOST_EXPECT(ret.ec != std::errc(), false)) {
      // Compatible with old format that generates very long mantissa from std stream.
      f = std::strtof(beg, nullptr);
    }
    return Json(static_cast<Number::Float>(f));
  } else {
    if (negative) {
      i = -i;
    }
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
    GetConsecutiveChar('r');
    GetConsecutiveChar('u');
    GetConsecutiveChar('e');
    result = true;
  } else {
    GetConsecutiveChar('a');
    GetConsecutiveChar('l');
    GetConsecutiveChar('s');
    GetConsecutiveChar('e');
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

void Json::Dump(Json json, std::string* str) {
  std::vector<char> buffer;
  JsonWriter writer(&buffer);
  writer.Save(json);
  str->resize(buffer.size());
  std::copy(buffer.cbegin(), buffer.cend(), str->begin());
}

Json& Json::operator=(Json const &other) = default;
}  // namespace xgboost
