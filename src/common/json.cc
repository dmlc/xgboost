/*!
 * Copyright (c) by Contributors 2019-2022
 */
#include "xgboost/json.h"

#include <dmlc/endian.h>

#include <cctype>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <sstream>

#include "./math.h"
#include "charconv.h"
#include "xgboost/base.h"
#include "xgboost/json_io.h"
#include "xgboost/logging.h"
#include "xgboost/string_view.h"

namespace xgboost {

void JsonWriter::Save(Json json) { json.Ptr()->Save(this); }

void JsonWriter::Visit(JsonArray const* arr) {
  this->WriteArray(arr, [](auto const& v) { return v; });
}
void JsonWriter::Visit(F32Array const* arr) {
  this->WriteArray(arr, [](float v) { return Json{v}; });
}
namespace {
auto to_i64 = [](auto v) { return Json{static_cast<int64_t>(v)}; };
}  // anonymous namespace
void JsonWriter::Visit(U8Array const* arr) { this->WriteArray(arr, to_i64); }
void JsonWriter::Visit(I32Array const* arr) { this->WriteArray(arr, to_i64); }
void JsonWriter::Visit(I64Array const* arr) { this->WriteArray(arr, to_i64); }

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
    case ValueKind::kString:
      return "String";
    case ValueKind::kNumber:
      return "Number";
    case ValueKind::kObject:
      return "Object";
    case ValueKind::kArray:
      return "Array";
    case ValueKind::kBoolean:
      return "Boolean";
    case ValueKind::kNull:
      return "Null";
    case ValueKind::kInteger:
      return "Integer";
    case ValueKind::kNumberArray:
      return "F32Array";
    case ValueKind::kU8Array:
      return "U8Array";
    case ValueKind::kI32Array:
      return "I32Array";
    case ValueKind::kI64Array:
      return "I64Array";
  }
  return "";
}

// Only used for keeping old compilers happy about non-reaching return
// statement.
Json& DummyJsonObject() {
  static Json obj;
  return obj;
}

Json& Value::operator[](std::string const&) {
  LOG(FATAL) << "Object of type " << TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& Value::operator[](int) {
  LOG(FATAL) << "Object of type " << TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

// Json Object
JsonObject::JsonObject(JsonObject&& that) noexcept : Value(ValueKind::kObject) {
  std::swap(that.object_, this->object_);
}

JsonObject::JsonObject(Map&& object) noexcept
    : Value(ValueKind::kObject), object_{std::forward<Map>(object)} {}

bool JsonObject::operator==(Value const& rhs) const {
  if (!IsA<JsonObject>(&rhs)) {
    return false;
  }
  return object_ == Cast<JsonObject const>(&rhs)->GetObject();
}

void JsonObject::Save(JsonWriter* writer) const { writer->Visit(this); }

// Json String
bool JsonString::operator==(Value const& rhs) const {
  if (!IsA<JsonString>(&rhs)) { return false; }
  return Cast<JsonString const>(&rhs)->GetString() == str_;
}

// FIXME: UTF-8 parsing support.
void JsonString::Save(JsonWriter* writer) const { writer->Visit(this); }

// Json Array
JsonArray::JsonArray(JsonArray&& that) noexcept : Value(ValueKind::kArray) {
  std::swap(that.vec_, this->vec_);
}

bool JsonArray::operator==(Value const& rhs) const {
  if (!IsA<JsonArray>(&rhs)) {
    return false;
  }
  auto& arr = Cast<JsonArray const>(&rhs)->GetArray();
  if (vec_.size() != arr.size()) {
    return false;
  }
  return std::equal(arr.cbegin(), arr.cend(), vec_.cbegin());
}

void JsonArray::Save(JsonWriter* writer) const { writer->Visit(this); }

// typed array
namespace {
// error C2668: 'fpclassify': ambiguous call to overloaded function
template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, bool> IsInfMSVCWar(T v) {
  return std::isinf(v);
}
template <typename T>
std::enable_if_t<std::is_integral<T>::value, bool> IsInfMSVCWar(T) {
  return false;
}
}  // namespace

template <typename T, Value::ValueKind kind>
void JsonTypedArray<T, kind>::Save(JsonWriter* writer) const {
  writer->Visit(this);
}

template <typename T, Value::ValueKind kind>
bool JsonTypedArray<T, kind>::operator==(Value const& rhs) const {
  if (!IsA<JsonTypedArray<T, kind>>(&rhs)) {
    return false;
  }
  auto& arr = Cast<JsonTypedArray<T, kind> const>(&rhs)->GetArray();
  if (vec_.size() != arr.size()) {
    return false;
  }
  if (std::is_same<float, T>::value) {
    for (size_t i = 0; i < vec_.size(); ++i) {
      bool equal{false};
      if (common::CheckNAN(vec_[i])) {
        equal = common::CheckNAN(arr[i]);
      } else if (IsInfMSVCWar(vec_[i])) {
        equal = IsInfMSVCWar(arr[i]);
      } else {
        equal = (arr[i] - vec_[i] == 0);
      }
      if (!equal) {
        return false;
      }
    }
    return true;
  }
  return std::equal(arr.cbegin(), arr.cend(), vec_.cbegin());
}

template class JsonTypedArray<float, Value::ValueKind::kNumberArray>;
template class JsonTypedArray<uint8_t, Value::ValueKind::kU8Array>;
template class JsonTypedArray<int32_t, Value::ValueKind::kI32Array>;
template class JsonTypedArray<int64_t, Value::ValueKind::kI64Array>;

// Json Number
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

void JsonNumber::Save(JsonWriter* writer) const { writer->Visit(this); }

// Json Integer
bool JsonInteger::operator==(Value const& rhs) const {
  if (!IsA<JsonInteger>(&rhs)) { return false; }
  return integer_ == Cast<JsonInteger const>(&rhs)->GetInteger();
}

void JsonInteger::Save(JsonWriter* writer) const { writer->Visit(this); }

// Json Null
bool JsonNull::operator==(Value const& rhs) const {
  if (!IsA<JsonNull>(&rhs)) { return false; }
  return true;
}

void JsonNull::Save(JsonWriter* writer) const { writer->Visit(this); }

// Json Boolean
bool JsonBoolean::operator==(Value const& rhs) const {
  if (!IsA<JsonBoolean>(&rhs)) { return false; }
  return boolean_ == Cast<JsonBoolean const>(&rhs)->GetBoolean();
}

void JsonBoolean::Save(JsonWriter* writer) const { writer->Visit(this); }

size_t constexpr JsonReader::kMaxNumLength;

Json JsonReader::Parse() {
  while (true) {
    SkipSpaces();
    auto c = PeekNextChar();
    if (c == -1) { break; }

    if (c == '{') {
      return ParseObject();
    } else if ( c == '[' ) {
      return ParseArray();
    } else if ( c == '-' || std::isdigit(c) ||
                c == 'N' || c == 'I') {
      // For now we only accept `NaN`, not `nan` as the later violates LR(1) with `null`.
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
  return {};
}

Json JsonReader::Load() {
  Json result = Parse();
  return result;
}

void JsonReader::Error(std::string msg) const {
  // just copy it.
  std::stringstream str_s;
  str_s << raw_str_.substr(0, raw_str_.size());

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

  auto raw_portion = raw_str_.substr(beg, end - beg);
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
bool IsSpace(JsonReader::Char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
}  // anonymous namespace

// Json class
void JsonReader::SkipSpaces() {
  while (cursor_.Pos() < raw_str_.size()) {
    Char c = raw_str_[cursor_.Pos()];
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
  Char ch { GetConsecutiveChar('\"') };  // NOLINT
  std::string str;
  while (true) {
    ch = GetNextChar();
    if (ch == '\\') {
      Char next{GetNextChar()};
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
  Char ch = GetNextNonSpaceChar();
  std::string buffer{static_cast<char>(ch)};
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

  Char ch { GetConsecutiveChar('[') };  // NOLINT
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

  Object::Map data;
  SkipSpaces();
  auto ch = PeekNextChar();

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
  Char ch = GetNextNonSpaceChar();
  std::string const t_value = u8"true";
  std::string const f_value = u8"false";

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

Json Json::Load(StringView str, std::ios::openmode mode) {
  Json json;
  if (mode & std::ios::binary) {
    UBJReader reader{str};
    json = Json::Load(&reader);
  } else {
    JsonReader reader(str);
    json = reader.Load();
  }
  return json;
}

Json Json::Load(JsonReader* reader) {
  Json json{reader->Load()};
  return json;
}

void Json::Dump(Json json, std::string* str, std::ios::openmode mode) {
  std::vector<char> buffer;
  Dump(json, &buffer, mode);
  str->resize(buffer.size());
  std::copy(buffer.cbegin(), buffer.cend(), str->begin());
}

void Json::Dump(Json json, std::vector<char>* str, std::ios::openmode mode) {
  str->clear();
  if (mode & std::ios::binary) {
    UBJWriter writer{str};
    writer.Save(json);
  } else {
    JsonWriter writer(str);
    writer.Save(json);
  }
}

void Json::Dump(Json json, JsonWriter* writer) {
  writer->Save(json);
}

static_assert(std::is_nothrow_move_constructible<Json>::value);
static_assert(std::is_nothrow_move_constructible<Object>::value);
static_assert(std::is_nothrow_move_constructible<Array>::value);
static_assert(std::is_nothrow_move_constructible<String>::value);

Json UBJReader::ParseArray() {
  auto marker = PeekNextChar();

  if (marker == '$') {  // typed array
    GetNextChar();      // remove $
    marker = GetNextChar();
    auto type = marker;
    GetConsecutiveChar('#');
    GetConsecutiveChar('L');
    auto n = this->ReadPrimitive<int64_t>();

    marker = PeekNextChar();
    switch (type) {
      case 'd':
        return ParseTypedArray<F32Array>(n);
      case 'U':
        return ParseTypedArray<U8Array>(n);
      case 'l':
        return ParseTypedArray<I32Array>(n);
      case 'L':
        return ParseTypedArray<I64Array>(n);
      default:
        LOG(FATAL) << "`" + std::string{static_cast<char>(type)} +  // NOLINT
                          "` is not supported for typed array.";
    }
  }
  std::vector<Json> results;
  if (marker == '#') {  // array with length optimization
    GetNextChar();
    GetConsecutiveChar('L');
    auto n = this->ReadPrimitive<int64_t>();
    results.resize(n);
    for (int64_t i = 0; i < n; ++i) {
      results[i] = Parse();
    }
  } else {  // normal array
    while (marker != ']') {
      results.emplace_back(Parse());
      marker = PeekNextChar();
    }
    GetConsecutiveChar(']');
  }

  return Json{results};
}

std::string UBJReader::DecodeStr() {
  // only L is supported right now.
  GetConsecutiveChar('L');
  auto bsize = this->ReadPrimitive<int64_t>();

  std::string str;
  str.resize(bsize);
  auto ptr = raw_str_.c_str() + cursor_.Pos();
  std::memcpy(&str[0], ptr, bsize);
  this->cursor_.Forward(bsize);
  return str;
}

Json UBJReader::ParseObject() {
  auto marker = PeekNextChar();
  Object::Map results;

  while (marker != '}') {
    auto str = this->DecodeStr();
    results.emplace(str, this->Parse());
    marker = PeekNextChar();
  }

  GetConsecutiveChar('}');
  return Json{std::move(results)};
}

Json UBJReader::Load() {
  Json result = Parse();
  return result;
}

Json UBJReader::Parse() {
  while (true) {
    auto c = PeekNextChar();
    if (c == -1) {
      break;
    }

    GetNextChar();
    switch (c) {
      case '{':
        return ParseObject();
      case '[':
        return ParseArray();
      case 'Z': {
        return Json{nullptr};
      }
      case 'T': {
        return Json{JsonBoolean{true}};
      }
      case 'F': {
        return Json{JsonBoolean{true}};
      }
      case 'd': {
        auto v = this->ReadPrimitive<float>();
        return Json{v};
      }
      case 'S': {
        auto str = this->DecodeStr();
        return Json{str};
      }
      case 'i': {
        Integer::Int i = this->ReadPrimitive<int8_t>();
        return Json{i};
      }
      case 'U': {
        Integer::Int i = this->ReadPrimitive<uint8_t>();
        return Json{i};
      }
      case 'I': {
        Integer::Int i = this->ReadPrimitive<int16_t>();
        return Json{i};
      }
      case 'l': {
        Integer::Int i = this->ReadPrimitive<int32_t>();
        return Json{i};
      }
      case 'L': {
        auto i = this->ReadPrimitive<int64_t>();
        return Json{i};
      }
      case 'C': {
        Integer::Int i = this->ReadPrimitive<char>();
        return Json{i};
      }
      case 'D': {
        LOG(FATAL) << "f64 is not supported.";
        break;
      }
      case 'H': {
        LOG(FATAL) << "High precision number is not supported.";
        break;
      }
      default:
        Error("Unknown construct");
    }
  }
  return {};
}

namespace {
template <typename T>
void WritePrimitive(T v, std::vector<char>* stream) {
  v = ToBigEndian(v);
  auto s = stream->size();
  stream->resize(s + sizeof(v));
  auto ptr = stream->data() + s;
  std::memcpy(ptr, &v, sizeof(v));
}

void EncodeStr(std::vector<char>* stream, std::string const& string) {
  stream->push_back('L');

  int64_t bsize = string.size();
  WritePrimitive(bsize, stream);

  auto s = stream->size();
  stream->resize(s + string.size());

  auto ptr = stream->data() + s;
  std::memcpy(ptr, string.data(), string.size());
}
}  // anonymous namespace

void UBJWriter::Visit(JsonArray const* arr) {
  stream_->emplace_back('[');
  auto const& vec = arr->GetArray();
  int64_t n = vec.size();
  stream_->push_back('#');
  stream_->push_back('L');
  WritePrimitive(n, stream_);
  for (auto const& v : vec) {
    this->Save(v);
  }
}

template <typename T, Value::ValueKind kind>
void WriteTypedArray(JsonTypedArray<T, kind> const* arr, std::vector<char>* stream) {
  stream->emplace_back('[');
  stream->push_back('$');
  if (std::is_same<T, float>::value) {
    stream->push_back('d');
  } else if (std::is_same<T, int8_t>::value) {
    stream->push_back('i');
  } else if (std::is_same<T, uint8_t>::value) {
    stream->push_back('U');
  } else if (std::is_same<T, int32_t>::value) {
    stream->push_back('l');
  } else if (std::is_same<T, int64_t>::value) {
    stream->push_back('L');
  } else {
    LOG(FATAL) << "Not implemented";
  }

  stream->push_back('#');
  stream->push_back('L');

  int64_t n = arr->Size();
  WritePrimitive(n, stream);
  auto s = stream->size();
  stream->resize(s + arr->Size() * sizeof(T));
  auto const& vec = arr->GetArray();
  for (int64_t i = 0; i < n; ++i) {
    auto v = ToBigEndian(vec[i]);
    std::memcpy(stream->data() + s, &v, sizeof(v));
    s += sizeof(v);
  }
}

void UBJWriter::Visit(F32Array const* arr) { WriteTypedArray(arr, stream_); }
void UBJWriter::Visit(U8Array const* arr) { WriteTypedArray(arr, stream_); }
void UBJWriter::Visit(I32Array const* arr) { WriteTypedArray(arr, stream_); }
void UBJWriter::Visit(I64Array const* arr) { WriteTypedArray(arr, stream_); }

void UBJWriter::Visit(JsonObject const* obj) {
  stream_->emplace_back('{');
  for (auto const& value : obj->GetObject()) {
    auto const& key = value.first;
    EncodeStr(stream_, key);
    this->Save(value.second);
  }
  stream_->emplace_back('}');
}

void UBJWriter::Visit(JsonNumber const* num) {
  stream_->push_back('d');
  auto val = num->GetNumber();
  WritePrimitive(val, stream_);
}

void UBJWriter::Visit(JsonInteger const* num) {
  auto i = num->GetInteger();
  if (i > std::numeric_limits<int8_t>::min() && i < std::numeric_limits<int8_t>::max()) {
    stream_->push_back('i');
    WritePrimitive(static_cast<int8_t>(i), stream_);
  } else if (i > std::numeric_limits<int16_t>::min() && i < std::numeric_limits<int16_t>::max()) {
    stream_->push_back('I');
    WritePrimitive(static_cast<int16_t>(i), stream_);
  } else if (i > std::numeric_limits<int32_t>::min() && i < std::numeric_limits<int32_t>::max()) {
    stream_->push_back('l');
    WritePrimitive(static_cast<int32_t>(i), stream_);
  } else {
    stream_->push_back('L');
    WritePrimitive(i, stream_);
  }
}

void UBJWriter::Visit(JsonNull const*) { stream_->push_back('Z'); }

void UBJWriter::Visit(JsonString const* str) {
  stream_->push_back('S');
  EncodeStr(stream_, str->GetString());
}

void UBJWriter::Visit(JsonBoolean const* boolean) {
  stream_->push_back(boolean->GetBoolean() ? 'T' : 'F');
}

void UBJWriter::Save(Json json) { json.Ptr()->Save(this); }
}  // namespace xgboost
