/*!
 * Copyright 2014 by Contributors
 * \file config.h
 * \brief helper class to load in configures from file
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_CONFIG_H_
#define XGBOOST_COMMON_CONFIG_H_

#include <cstdio>
#include <cstring>
#include <string>
#include <istream>
#include <fstream>

namespace xgboost {
namespace common {
/*!
 * \brief base implementation of config reader
 */
class ConfigReaderBase {
 public:
  /*!
   * \brief get current name, called after Next returns true
   * \return current parameter name
   */
  explicit ConfigParser(const std::string& path)
      : path_(path),
      line_comment_regex_("^#"),
      key_regex_(R"rx(^([^#"'=\r\n\t ]+)[\t ]*=)rx"),
      key_regex_escaped_(R"rx(^(["'])([^"'=\r\n]+)\1[\t ]*=)rx"),
      value_regex_(R"rx(^([^#"'=\r\n\t ]+)[\t ]*(?:#.*){0,1}$)rx"),
      value_regex_escaped_(R"rx(^(["'])([^"'=\r\n]+)\1[\t ]*(?:#.*){0,1}$)rx")
  {}

  std::string LoadConfigFile(const std::string& path) {
    std::ifstream fin(path, std::ios_base::in | std::ios_base::binary);
    CHECK(fin) << "Failed to open: " << path;
    std::string content{std::istreambuf_iterator<char>(fin),
                        std::istreambuf_iterator<char>()};
    return content;
  }
  /*!
   * \brief get current value, called after Next returns true
   * \return current parameter value
   */
  inline const char *Val() const {
    return s_val_.c_str();
  }
  /*!
   * \brief move iterator to next position
   * \return true if there is value in next position
   */
  std::vector<std::pair<std::string, std::string>> Parse() {
    std::string content { LoadConfigFile(path_) };
    content = NormalizeConfigEOL(content);
    std::stringstream ss { content };
    std::vector<std::pair<std::string, std::string>> results;
    std::string line;
    std::string key, value;
    // Loop over every line of the configuration file
    while (std::getline(ss, line)) {
      if (ParseKeyValuePair(line, &key, &value)) {
        results.emplace_back(key, value);
      }
    }
    return false;
  }
  // called before usage
  inline void Init() {
    ch_buf_ = this->GetChar();
  }

 protected:
  /*!
   * \brief to be implemented by subclass,
   * get next token, return EOF if end of file
   */
  virtual int GetChar() = 0;
  /*! \brief to be implemented by child, check if end of stream */
  virtual bool IsEnd() = 0;

 private:
  int ch_buf_;
  std::string s_name_, s_val_, s_buf_;

  inline void SkipLine() {
    do {
      ch_buf_ = this->GetChar();
    } while (ch_buf_ != EOF && ch_buf_ != '\n' && ch_buf_ != '\r');
  }

  inline void ParseStr(std::string *tok) {
    while ((ch_buf_ = this->GetChar()) != EOF) {
      switch (ch_buf_) {
        case '\\': *tok += this->GetChar(); break;
        case '\"': return;
        case '\r':
        case '\n': LOG(FATAL)<< "ConfigReader: unterminated string";
        default: *tok += static_cast<char>(ch_buf_);
      }
    }
    LOG(FATAL) << "ConfigReader: unterminated string";
  }
  inline void ParseStrML(std::string *tok) {
    while ((ch_buf_ = this->GetChar()) != EOF) {
      switch (ch_buf_) {
        case '\\': *tok += this->GetChar(); break;
        case '\'': return;
        default: *tok += static_cast<char>(ch_buf_);
      }
    }
    LOG(FATAL) << "unterminated string";
  }
  // return newline
  inline bool GetNextToken(std::string *tok) {
    tok->clear();
    bool new_line = false;
    while (ch_buf_ != EOF) {
      switch (ch_buf_) {
        case '#' : SkipLine(); new_line = true; break;
        case '\"':
          if (tok->length() == 0) {
            ParseStr(tok); ch_buf_ = this->GetChar(); return new_line;
          } else {
            LOG(FATAL) << "ConfigReader: token followed directly by string";
          }
        case '\'':
          if (tok->length() == 0) {
            ParseStrML(tok); ch_buf_ = this->GetChar(); return new_line;
          } else {
            LOG(FATAL) << "ConfigReader: token followed directly by string";
          }
        case '=':
          if (tok->length() == 0) {
            ch_buf_ = this->GetChar();
            *tok = '=';
          }
          return new_line;
        case '\r':
        case '\n':
          if (tok->length() == 0) new_line = true;
        case '\t':
        case ' ' :
          ch_buf_ = this->GetChar();
          if (tok->length() != 0) return new_line;
          break;
        default:
          *tok += static_cast<char>(ch_buf_);
          ch_buf_ = this->GetChar();
          break;
      }
    }
    if (tok->length() == 0) {
      return true;
    } else {
      return false;
    }
  }
};
/*!
 * \brief an iterator use stream base, allows use all types of istream
 */
class ConfigStreamReader: public ConfigReaderBase {
 public:
  /*!
   * \brief constructor
   * \param fin istream input stream
   */
  explicit ConfigStreamReader(std::istream &fin) : fin_(fin) {}

 protected:
  int GetChar() override {
    return fin_.get();
  }
  /*! \brief to be implemented by child, check if end of stream */
  bool IsEnd() override {
    return fin_.eof();
  }

 private:
  std::istream &fin_;
};

/*!
 * \brief an iterator that iterates over a configure file and gets the configures
 */
class ConfigIterator: public ConfigStreamReader {
 public:
  /*!
   * \brief constructor
   * \param fname name of configure file
   */
  explicit ConfigIterator(const char *fname) : ConfigStreamReader(fi_) {
    fi_.open(fname);
    if (fi_.fail()) {
      LOG(FATAL) << "cannot open file " << fname;
    }
    ConfigReaderBase::Init();
  }
  /*! \brief destructor */
  ~ConfigIterator() {
    fi_.close();
  }

 private:
  std::ifstream fi_;
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_CONFIG_H_
