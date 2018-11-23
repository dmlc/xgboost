/*!
 * Copyright 2018 by Contributors
 * \file json.cc
 * \brief Simple implementation of JSON.
 */
#include <dmlc/logging.h>
#include <xgboost/base.h>

#include <locale>
#include <cctype>  // isdigit, isspace
#include "json.h"

#if defined(_MSC_VER)
#define u8
#endif

namespace xgboost {
namespace json {

class JsonWriter {
  static constexpr size_t kIndentSize = 2;

  size_t n_spaces_;
  std::ostream* stream_;

  std::locale original_locale_;

 public:
  explicit JsonWriter(std::ostream* stream) : n_spaces_{0}, stream_{stream} {
    original_locale_ = std::locale("");
    stream_->imbue(std::locale("en_US.UTF-8"));
  }
  ~JsonWriter() {
    stream_->imbue(original_locale_);
  }

  void NewLine() {
    *stream_ << u8"\n" << std::string(n_spaces_, ' ');
  }

  void BeginIndent() {
    n_spaces_ += kIndentSize;
  }
  void EndIndent() {
    n_spaces_ -= kIndentSize;
    CHECK_GE(n_spaces_, 0);
  }

  void Write(std::string str) {
    *stream_ << str;
  }

  void Save(Json json) {
    json.ptr_->Save(this);
  }
};

class JsonReader {
 private:
  struct SourceLocation {
    int cl_;      // current line
    int cc_;      // current column
    size_t pos_;  // current position in raw_str_

   public:
    SourceLocation() : cl_(0), cc_(0), pos_(0) {}

    int Line() const { return cl_;  }
    int Col()  const { return cc_;  }
    size_t Pos()  const { return pos_; }

    SourceLocation& Forward(char c = 0) {
      if (c == '\n') {
        cc_ = 0;
        cl_++;
      } else {
        cc_++;
      }
      pos_++;
      return *this;
    }
  } cursor_;

  std::string raw_str_;

 private:
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
    if (result != c) { Expect(c); }
    return result;
  }

  void Error(std::string msg) const {
    std::istringstream str_s(raw_str_);

    msg += ", at ("
           + std::to_string(cursor_.Line()) + ", "
           + std::to_string(cursor_.Col()) + ")\n";
    std::string line;
    int line_count = 0;
    while (std::getline(str_s, line) && line_count < cursor_.Line()) {
      line_count++;
    }
    msg+= line += '\n';
    std::string spaces(cursor_.Col(), ' ');
    msg+= spaces + "^\n";

    LOG(FATAL) << msg;
  }

  // Report expected character
  void Expect(char c) {
    std::string msg = "Expecting: \"";
    msg += std::string {c}
           + "\", got: \"" + raw_str_[cursor_.Pos()-1] + "\"\n";
    Error(msg);
  }

  Json ParseString();
  Json ParseObject();
  Json ParseArray();
  Json ParseNumber();
  Json ParseBoolean();

  Json Parse() {
    while (true) {
      SkipSpaces();
      char c = PeekNextChar();
      if (c == -1) { break; }

      if (c == '{') {
        return ParseObject();
      } else if ( c == '[' ) {
        return ParseArray();
      } else if ( c == '-' || std::isdigit(c) ) {
        return ParseNumber();
      } else if ( c == '\"' ) {
        return ParseString();
      } else if ( c == 't' || c == 'f' ) {
        return ParseBoolean();
      } else {
        Error("Unknown construct");
      }
    }
    return Json();
  }

 private:
  std::locale original_locale_;
  std::istream* stream_;

 public:
  explicit JsonReader(std::istream* stream) {
    original_locale_ = std::locale("");
    stream_ = stream;
    stream->imbue(std::locale("en_US.UTF-8"));
  }
  ~JsonReader() {
    stream_->imbue(original_locale_);
  }

  Json Load() {
    raw_str_ = std::string(std::istreambuf_iterator<char>(*stream_), {});
    Json result = Parse();
    stream_->imbue(original_locale_);
    return result;
  }
};

// Value
std::string Value::TypeStr() const {
  switch (kind_) {
    case ValueKind::kString: return "String";  break;
    case ValueKind::kNumber: return "Number";  break;
    case ValueKind::kObject: return "Object";  break;
    case ValueKind::kArray:  return "Array";   break;
    case ValueKind::kBoolean:return "Boolean"; break;
    case ValueKind::kNull:   return "Null";    break;
  }
  return "";
}

// Json Object
JsonObject::JsonObject() : Value(ValueKind::kObject) {}
JsonObject::JsonObject(std::map<std::string, Json>&& object)
    : Value(ValueKind::kObject), object_(std::move(object)) {}
JsonObject::JsonObject(std::map<std::string, Json> const& object)
    : Value(ValueKind::kObject), object_(object) {}

Json& JsonObject::operator[](std::string const& key) {
  return object_[key];
}

// Only used for keeping old compilers happy about non-reaching return
// statement.
Json& DummyJsonObject() {
  static Json obj;
  return obj;
}

// object
Json& JsonObject::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr()
             << " can not be indexed by integer.";
  return DummyJsonObject();
}

bool JsonObject::operator==(Value const& rhs) const {
  if ( !IsA<JsonObject>(&rhs) ) { return false; }
  if (object_.size() != Cast<JsonObject const>(&rhs)->GetObject().size()) {
    return false;
  }
  bool result = object_ == Cast<JsonObject const>(&rhs)->GetObject();
  return result;
}

void JsonObject::Save(JsonWriter* writer) {
  writer->Write("{");
  writer->BeginIndent();
  writer->NewLine();

  size_t i = 0;
  size_t size = object_.size();

  for (auto& value : object_) {
    writer->Write("\"" + value.first + "\": ");
    writer->Save(value.second);

    if (i != size-1) {
      writer->Write(",");
      writer->NewLine();
    }
    i++;
  }
  writer->EndIndent();
  writer->NewLine();
  writer->Write("}");
}

// Json String
Json& JsonString::operator[](std::string const & key) {
  throw std::runtime_error(
      "Object of type " +
      Value::TypeStr() + " can not be indexed by string.");
  return DummyJsonObject();
}

Json& JsonString::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr()
             << " can not be indexed by integer."
             << "please try obtaining std::string first.";
  return DummyJsonObject();
}

bool JsonString::operator==(Value const& rhs) const {
  if (!IsA<JsonString>(&rhs)) { return false; }
  return Cast<JsonString const>(&rhs)->GetString() == str_;
}

// FIXME: UTF-8 parsing support.
void JsonString::Save(JsonWriter* writer) {
  std::string buffer;
  buffer += '"';
  for (size_t i = 0; i < str_.length(); i++) {
    const char ch = str_[i];
    if (ch == '\\') {
      if (i < str_.size() && str_[i+1] == 'u') {
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
  writer->Write(buffer);
}

// Json Array
JsonArray::JsonArray() : Value(ValueKind::kArray) {}
JsonArray::JsonArray(std::vector<Json>&& arr) :
    Value(ValueKind::kArray), vec_(std::move(arr)) {}
JsonArray::JsonArray(std::vector<Json> const& arr) :
    Value(ValueKind::kArray), vec_(arr) {}

Json& JsonArray::operator[](std::string const & key) {
  throw std::runtime_error(
      "Object of type " +
      Value::TypeStr() + " can not be indexed by string.");
  return DummyJsonObject();
}

Json& JsonArray::operator[](int ind) {
  return vec_.at(ind);
}

bool JsonArray::operator==(Value const& rhs) const {
  if (!IsA<JsonArray>(&rhs)) { return false; }
  auto& arr = Cast<JsonArray const>(&rhs)->GetArray();
  if (arr.size() != vec_.size()) { return false; }
  return std::equal(arr.cbegin(), arr.cend(), vec_.cbegin());
}

void JsonArray::Save(JsonWriter* writer) {
  writer->Write("[");
  size_t size = vec_.size();
  for (size_t i = 0; i < size; ++i) {
    auto& value = vec_[i];
    writer->Save(value);
    if (i != size-1) { writer->Write(", "); }
  }
  writer->Write("]");
}

// Json Number
JsonNumber::JsonNumber(double value) :  // NOLINT
    Value(ValueKind::kNumber), number_(value) {}
Json& JsonNumber::operator[](std::string const & key) {
  throw std::runtime_error(
      "Object of type " +
      Value::TypeStr() + " can not be indexed by string.");
  return DummyJsonObject();
}

Json& JsonNumber::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr()
             << " can not be indexed by integer.";
  return DummyJsonObject();
}

bool JsonNumber::operator==(Value const& rhs) const {
  if (!IsA<JsonNumber>(&rhs)) { return false; }
  double residue =
      std::abs(number_ - Cast<JsonNumber const>(&rhs)->GetDouble());
  return residue < kRtEps;
}

void JsonNumber::Save(JsonWriter* writer) {
  writer->Write(std::to_string(this->GetDouble()));
}

double JsonNumber::GetDouble()  const { return number_; }
int    JsonNumber::GetInteger() const {
  return static_cast<int>(number_);
}
float  JsonNumber::GetFloat()   const {
  return static_cast<float>(number_);
}

// Json Null
Json& JsonNull::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr()
             << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonNull::operator[](int ind) {
  throw std::runtime_error(
      "Object of type " +
      Value::TypeStr() + " can not be indexed by integer.");
  return DummyJsonObject();
}

bool JsonNull::operator==(Value const& rhs) const {
  if ( !IsA<JsonNull>(&rhs) ) { return false; }
  return true;
}

void JsonNull::Save(JsonWriter* writer) {
  LOG(FATAL) << "Saving null";
  writer->Write("null");
}

// Json Boolean
Json& JsonBoolean::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr()
             << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonBoolean::operator[](int ind) {
  throw std::runtime_error(
      "Object of type " +
      Value::TypeStr() + " can not be indexed by Integer.");
  return DummyJsonObject();
}

bool JsonBoolean::operator==(Value const& rhs) const {
  if (!IsA<JsonBoolean>(&rhs)) { return false; }
  return boolean_ == Cast<JsonBoolean const>(&rhs)->GetBoolean();
}

void JsonBoolean::Save(JsonWriter *writer) {
  if (boolean_) {
    writer->Write(u8"true");
  } else {
    writer->Write(u8"false");
  }
}

// Json class
void JsonReader::SkipSpaces() {
  while (cursor_.Pos() < raw_str_.size()) {
    char c = raw_str_[cursor_.Pos()];
    if (std::isspace(c)) {
      cursor_.Forward(c);
    } else {
      break;
    }
  }
}

Json JsonReader::ParseString() {
  GetChar('\"');
  std::ostringstream output;
  std::string str;
  while (true) {
    char ch = GetNextChar();
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
      Expect('\"');
    }
  }
  return Json(std::move(str));
}

Json JsonReader::ParseArray() {
  std::vector<Json> data;

  GetChar('[');
  while (true) {
    if (PeekNextChar() == ']') {
      GetChar(']');
      return Json(std::move(data));
    }
    auto obj = Parse();
    data.push_back(obj);
    char ch = GetNextNonSpaceChar();
    if (ch == ']') break;
    if (ch != ',') {
      Expect(',');
    }
  }

  return Json(std::move(data));
}

Json JsonReader::ParseObject() {
  char ch = GetChar('{');
  SkipSpaces();

  std::map<std::string, Json> data;
  if (PeekNextChar() == '}') {
    GetNextNonSpaceChar();
    return Json(std::move(data));
  }

  while (true) {
    SkipSpaces();
    ch = PeekNextChar();
    if (ch != '"') {
      Expect('"');
    }
    Json key = ParseString();

    ch = GetNextNonSpaceChar();

    if (ch != ':') {
      Expect(':');
    }

    Json value {Parse()};

    data[Cast<JsonString>(&(key.GetValue()))->GetString()] = std::move(value);

    ch = GetNextNonSpaceChar();

    if (ch == '}') break;
    if (ch != ',') {
      Expect(',');
    }
  }

  return Json(std::move(data));
}

Json JsonReader::ParseNumber() {
  std::string substr = raw_str_.substr(cursor_.Pos(), 17);
  size_t pos = 0;
  double number = std::stod(substr, &pos);
  for (size_t i = 0; i < pos; ++i) {
    GetNextChar();
  }
  return Json(number);
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

Json Json::Load(std::istream* stream) {
  JsonReader reader(stream);
  try {
    Json j{reader.Load()};
    return j;
  } catch (std::runtime_error const& e) {
    std::cerr << e.what();
    return Json();
  }
}

void Json::Dump(Json j, std::ostream *stream) {
  JsonWriter writer(stream);
  try {
    writer.Save(j);
  } catch (dmlc::Error const& e) {
    std::cerr << e.what();
  }
}

}  // namespace json
}  // namespace xgboost

#if defined(_MSC_VER)
#undef u8
#endif
