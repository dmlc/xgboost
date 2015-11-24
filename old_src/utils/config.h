/*!
 * Copyright 2014 by Contributors
 * \file config.h
 * \brief helper class to load in configures from file
 * \author Tianqi Chen
 */
#ifndef XGBOOST_UTILS_CONFIG_H_
#define XGBOOST_UTILS_CONFIG_H_

#include <cstdio>
#include <cstring>
#include <string>
#include <istream>
#include <fstream>
#include "./utils.h"

namespace xgboost {
namespace utils {
/*!
 * \brief base implementation of config reader
 */
class ConfigReaderBase {
 public:
  /*!
   * \brief get current name, called after Next returns true
   * \return current parameter name
   */
  inline const char *name(void) const {
    return s_name.c_str();
  }
  /*!
   * \brief get current value, called after Next returns true
   * \return current parameter value
   */
  inline const char *val(void) const {
    return s_val.c_str();
  }
  /*!
   * \brief move iterator to next position
   * \return true if there is value in next position
   */
  inline bool Next(void) {
    while (!this->IsEnd()) {
      GetNextToken(&s_name);
      if (s_name == "=") return false;
      if (GetNextToken(&s_buf) || s_buf != "=")  return false;
      if (GetNextToken(&s_val) || s_val == "=")  return false;
      return true;
    }
    return false;
  }
  // called before usage
  inline void Init(void) {
    ch_buf = this->GetChar();
  }

 protected:
  /*!
   * \brief to be implemented by subclass,
   * get next token, return EOF if end of file
   */
  virtual char GetChar(void) = 0;
  /*! \brief to be implemented by child, check if end of stream */
  virtual bool IsEnd(void) = 0;

 private:
  char ch_buf;
  std::string s_name, s_val, s_buf;

  inline void SkipLine(void) {
    do {
      ch_buf = this->GetChar();
    } while (ch_buf != EOF && ch_buf != '\n' && ch_buf != '\r');
  }

  inline void ParseStr(std::string *tok) {
    while ((ch_buf = this->GetChar()) != EOF) {
      switch (ch_buf) {
        case '\\': *tok += this->GetChar(); break;
        case '\"': return;
        case '\r':
        case '\n': Error("ConfigReader: unterminated string");
        default: *tok += ch_buf;
      }
    }
    Error("ConfigReader: unterminated string");
  }
  inline void ParseStrML(std::string *tok) {
    while ((ch_buf = this->GetChar()) != EOF) {
      switch (ch_buf) {
        case '\\': *tok += this->GetChar(); break;
        case '\'': return;
        default: *tok += ch_buf;
      }
    }
    Error("unterminated string");
  }
  // return newline
  inline bool GetNextToken(std::string *tok) {
    tok->clear();
    bool new_line = false;
    while (ch_buf != EOF) {
      switch (ch_buf) {
        case '#' : SkipLine(); new_line = true; break;
        case '\"':
          if (tok->length() == 0) {
            ParseStr(tok); ch_buf = this->GetChar(); return new_line;
          } else {
            Error("ConfigReader: token followed directly by string");
          }
        case '\'':
          if (tok->length() == 0) {
            ParseStrML(tok); ch_buf = this->GetChar(); return new_line;
          } else {
            Error("ConfigReader: token followed directly by string");
          }
        case '=':
          if (tok->length() == 0) {
            ch_buf = this->GetChar();
            *tok = '=';
          }
          return new_line;
        case '\r':
        case '\n':
          if (tok->length() == 0) new_line = true;
        case '\t':
        case ' ' :
          ch_buf = this->GetChar();
          if (tok->length() != 0) return new_line;
          break;
        default:
          *tok += ch_buf;
          ch_buf = this->GetChar();
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
   * \param istream input stream
   */
  explicit ConfigStreamReader(std::istream &fin) : fin(fin) {}

 protected:
  virtual char GetChar(void) {
    return fin.get();
  }
  /*! \brief to be implemented by child, check if end of stream */
  virtual bool IsEnd(void) {
    return fin.eof();
  }

 private:
  std::istream &fin;
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
  explicit ConfigIterator(const char *fname) : ConfigStreamReader(fi) {
    fi.open(fname);
    if (fi.fail()) {
      utils::Error("cannot open file %s", fname);
    }
    ConfigReaderBase::Init();
  }
  /*! \brief destructor */
  ~ConfigIterator(void) {
    fi.close();
  }

 private:
  std::ifstream fi;
};
}  // namespace utils
}  // namespace xgboost
#endif  // XGBOOST_UTILS_CONFIG_H_
