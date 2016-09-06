/*!
 *  Copyright (c) 2015 by Contributors
 */
#include <sstream>
#include <exception>

#include "dmlc/config.h"
#include "dmlc/logging.h"

using namespace std;

namespace dmlc {

struct Token {
  std::string buf;
  bool is_string;
};

class TokenizeError : public exception {
 public:
  explicit TokenizeError(const string& msg = "tokenize error"): msg_(msg) { }
  ~TokenizeError() throw() {}
  virtual const char* what() const throw() {
    return msg_.c_str();
  }
 private:
  string msg_;
};

class Tokenizer {
 public:
  explicit Tokenizer(istream& is): is_(is), state_(kNone) {}  // NOLINT(*)
  bool GetNextToken(Token* tok) {
    // token is defined as
    // 1. [^\s=]+
    // 2. "[(^"|\\")]*"
    // 3. =
    state_ = kNone;
    tok->buf.clear();
    tok->is_string = false;
    char ch;
    while ( (ch = PeekChar()) != EOF && state_ != kFinish ) {
      switch (ch) {
      case ' ': case '\t': case '\n': case '\r':
        if (state_ == kToken) {
          state_ = kFinish;
        } else {
          EatChar();  // ignore
        }
        break;
      case '\"':
        ParseString(&tok->buf);
        state_ = kFinish;
        tok->is_string = true;
        break;
      case '=':
        if (state_ != kToken) {
          tok->buf = '=';
          EatChar();
        }
        state_ = kFinish;
        break;
      case '#':
        ParseComments();
        break;
      default:
        state_ = kToken;
        tok->buf += ch;
        EatChar();
        break;
      }
    }
    return PeekChar() != EOF;
  }

  void ParseString(string* tok) {
    EatChar();  // eat the first quotation mark
    char ch;
    while ( (ch = PeekChar()) != '\"' ) {
      switch (ch) {
        case '\\':
          EatChar();
          ch = PeekChar();
          if (ch == '\"') {
            *tok += '\"';
          } else {
            throw TokenizeError("error parsing escape characters");
          }
          break;
        case '\n': case '\r': case EOF:
          throw TokenizeError("quotation mark is not closed");
        default:
          *tok += ch;
          break;
      }
      EatChar();
    }
    EatChar();  // eat the last quotation mark
  }

  void ParseComments() {
    char ch;
    while ( (ch = PeekChar()) ) {
      if (ch == '\n' || ch == '\r' || ch == EOF) {
        break;  // end of comment
      }
      EatChar();  // ignore all others
    }
  }

 private:
  char PeekChar() {
    return is_.peek();
  }
  void EatChar() {
    is_.get();
  }

  enum ParseState {
    kNone = 0,
    kToken,
    kFinish,
  };
  istream& is_;
  ParseState state_;
};

//////////////////////// Config /////////////////////////////
Config::Config(bool m): multi_value_(m) {
  Clear();
}

Config::Config(istream& is, bool m): multi_value_(m) {
  Clear();
  LoadFromStream(is);
}

void Config::Clear() {
  config_map_.clear();
  order_.clear();
}

void Config::LoadFromStream(istream& is) {
  Tokenizer tokenizer(is);
  Token key, eqop, value;
  try {
    while ( true ) {
      tokenizer.GetNextToken(&key);
      if (key.buf.length() == 0) {
        break;  // no content left
      }
      tokenizer.GetNextToken(&eqop);
      tokenizer.GetNextToken(&value);
      if (eqop.buf != "=") {
        LOG(ERROR) << "Parsing error: expect format \"k = v\"; but got \""
          << key.buf << eqop.buf << value.buf << "\"";
      }
      Insert(key.buf, value.buf, value.is_string);
    }
  } catch(TokenizeError& err) {
    LOG(ERROR) << "Tokenize error: " << err.what();
  }
}

const string& Config::GetParam(const string& key) const {
  CHECK(config_map_.find(key) != config_map_.end())
      << "key \"" << key << "\" not found in configure";
  const std::vector<std::string>& vals = config_map_.find(key)->second.val;
  return vals[vals.size() - 1];  // return tne latest inserted one
}

bool Config::IsGenuineString(const std::string& key) const {
  CHECK(config_map_.find(key) != config_map_.end())
      << "key \"" << key << "\" not found in configure";
  return config_map_.find(key)->second.is_string;
}

string MakeProtoStringValue(const std::string& str) {
  string rst = "\"";
  for (size_t i = 0; i < str.length(); ++i) {
    if (str[i] != '\"') {
      rst += str[i];
    } else {
      rst += "\\\"";
    }
  }
  rst += "\"";
  return rst;
}

string Config::ToProtoString(void) const {
  ostringstream oss;
  for (ConfigIterator iter = begin(); iter != end(); ++iter) {
    const ConfigEntry& entry = *iter;
    bool is_string = IsGenuineString(entry.first);
    oss << entry.first << " : " <<
      (is_string? MakeProtoStringValue(entry.second) : entry.second)
      << "\n";
  }
  return oss.str();
}

Config::ConfigIterator Config::begin() const {
  return ConfigIterator(0, this);
}

Config::ConfigIterator Config::end() const {
  return ConfigIterator(order_.size(), this);
}

void Config::Insert(const std::string& key, const std::string& value, bool is_string) {
  size_t insert_index = order_.size();
  if (!multi_value_) {
    config_map_[key] = ConfigValue();
  }
  ConfigValue& cv = config_map_[key];
  size_t val_index = cv.val.size();
  cv.val.push_back(value);
  cv.insert_index.push_back(insert_index);
  cv.is_string = is_string;

  order_.push_back(make_pair(key, val_index));
}

////////////////////// ConfigIterator //////////////////////

Config::ConfigIterator::ConfigIterator(size_t i, const Config* c)
    : index_(i), config_(c) {
  FindNextIndex();
}

Config::ConfigIterator::ConfigIterator(const Config::ConfigIterator& other)
    : index_(other.index_), config_(other.config_) {
}

Config::ConfigIterator& Config::ConfigIterator::operator++() {
  if (index_ < config_->order_.size()) {
    ++index_;
  }
  FindNextIndex();
  return *this;
}

Config::ConfigIterator Config::ConfigIterator::operator++(int any) {
  ConfigIterator tmp(*this);
  operator++();
  return tmp;
}

bool Config::ConfigIterator::operator==(const Config::ConfigIterator& rhs) const {
  return index_ == rhs.index_ && config_ == rhs.config_;
}

bool Config::ConfigIterator::operator!=(const Config::ConfigIterator& rhs) const {
  return !(operator == (rhs));
}

Config::ConfigEntry Config::ConfigIterator::operator * () const {
  const std::string& key = config_->order_[index_].first;
  size_t val_index = config_->order_[index_].second;
  const std::string& val = config_->config_map_.find(key)->second.val[val_index];
  return make_pair(key, val);
}

void Config::ConfigIterator::FindNextIndex() {
  bool found = false;
  while (!found && index_ < config_->order_.size()) {
    const std::string& key = config_->order_[index_].first;
    size_t val_index = config_->order_[index_].second;
    size_t val_insert_index = config_->config_map_.find(key)->second.insert_index[val_index];
    if (val_insert_index == index_) {
      found = true;
    } else {
      ++index_;
    }
  }
}

}  // namespace dmlc
