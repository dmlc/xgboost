/*!
 * Copyright (c) 2015 by Contributors
 * \file json.h
 * \brief Lightweight JSON Reader/Writer that read save into C++ data structs.
 *  This includes STL composites and structures.
 */
#ifndef DMLC_JSON_H_
#define DMLC_JSON_H_

// This code requires C++11 to compile
#include <vector>
#include <iostream>
#include <cctype>
#include <string>
#include <algorithm>
#include <map>
#include <list>
#include <utility>

#include "./base.h"
#include "./logging.h"
#include "./type_traits.h"

#if DMLC_USE_CXX11
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#if DMLC_STRICT_CXX11
#include "./any.h"
#endif  // DMLC_STRICT_CXX11
#endif  // DMLC_USE_CXX11

namespace dmlc {
/*!
 * \brief Lightweight JSON Reader to read any STL compositions and structs.
 *  The user need to know the schema of the
 *
 */
class JSONReader {
 public:
  /*!
   * \brief Constructor.
   * \param is the input stream.
   */
  explicit JSONReader(std::istream *is)
      : is_(is),
        line_count_r_(0),
        line_count_n_(0) {}
  /*!
   * \brief Parse next JSON string.
   * \param out_str the output string.
   * \throw dmlc::Error when next token is not string
   */
  inline void ReadString(std::string *out_str);
  /*!
   * \brief Read Number.
   * \param out_value output value;
   * \throw dmlc::Error when next token is not number of ValueType.
   * \tparam ValueType type of the number
   */
  template<typename ValueType>
  inline void ReadNumber(ValueType *out_value);
  /*!
   * \brief Begin parsing an object.
   * \code
   *  std::string key;
   *  // value can be any type that is json serializable.
   *  std::string value;
   *  reader->BeginObject();
   *  while (reader->NextObjectItem(&key)) {
   *    // do somthing to key value
   *    reader->Read(&value);
   *  }
   * \endcode
   */
  inline void BeginObject();
  /*!
   * \brief Begin parsing an array.
   * \code
   *  // value can be any type that is json serializable.
   *  std::string value;
   *  reader->BeginArray();
   *  while (reader->NextObjectArrayItem(&value)) {
   *    // do somthing to value
   *  }
   * \endcode
   */
  inline void BeginArray();
  /*!
   * \brief Try to move to next object item.
   *  If this call is successful, user can proceed to call
   *  reader->Read to read in the value.
   * \param out_key the key to the next object.
   * \return true if the read is successful, false if we are at end of the object.
   */
  inline bool NextObjectItem(std::string *out_key);
  /*!
   * \brief Try to read the next element in the array.
   *  If this call is successful, user can proceed to call
   *  reader->Read to read in the value.
   * \return true if the read is successful, false if we are at end of the array.
   */
  inline bool NextArrayItem();
  /*!
   * \brief Read next ValueType.
   * \param out_value any STL or json readable type to be read
   * \throw dmlc::Error when the read of ValueType is not successful.
   * \tparam ValueType the data type to be read.
   */
  template<typename ValueType>
  inline void Read(ValueType *out_value);

  /*! \return current line count */
  inline std::string line_info() const {
    char temp[64];
    std::ostringstream os;
    os << " Line " << std::max(line_count_r_, line_count_n_);
    is_->getline(temp, 64);
    os << ", around ^`" << temp << "`";
    return os.str();
  }

 private:
  /*! \brief internal reader stream */
  std::istream *is_;
  /*! \brief "\\r" counter */
  size_t line_count_r_;
  /*! \brief "\\n" counter */
  size_t line_count_n_;
  /*!
   * \brief record how many element processed in
   *  current array/object scope.
   */
  std::vector<size_t> scope_counter_;
  /*!
   * \brief Read next nonspace character.
   * \return the next nonspace character.
   */
  inline int NextNonSpace();
  /*!
   * \brief Read just before next nonspace but not read that.
   * \return the next nonspace character.
   */
  inline int PeekNextNonSpace();
};

/*!
 * \brief Lightweight json to write any STL compositions.
 */
class JSONWriter {
 public:
  /*!
   * \brief Constructor.
   * \param os the output stream.
   */
  explicit JSONWriter(std::ostream *os)
      : os_(os) {}
  /*!
   * \brief Write a string that do not contain escape characters.
   * \param s the string to be written.
   */
  inline void WriteNoEscape(const std::string &s);
  /*!
   * \brief Write a string that can contain escape characters.
   * \param s the string to be written.
   */
  inline void WriteString(const std::string &s);
  /*!
   * \brief Write a string that can contain escape characters.
   * \param v the value to be written.
   * \tparam ValueType The value type to be written.
   */
  template<typename ValueType>
  inline void WriteNumber(const ValueType &v);
  /*!
   * \brief Start beginning of array.
   * \param multi_line whether to start an multi_line array.
   * \code
   *  writer->BeginArray();
   *  for (auto& v : vdata) {
   *    writer->WriteArrayItem(v);
   *  }
   *  writer->EndArray();
   * \endcode
   */
  inline void BeginArray(bool multi_line = true);
  /*! \brief Finish writing an array. */
  inline void EndArray();
  /*!
   * \brief Start beginning of array.
   * \param multi_line whether to start an multi_line array.
   * \code
   *  writer->BeginObject();
   *  for (auto& kv : vmap) {
   *    writer->WriteObjectKeyValue(kv.first, kv.second);
   *  }
   *  writer->EndObject();
   * \endcode
   */
  inline void BeginObject(bool multi_line = true);
  /*! \brief Finish writing object. */
  inline void EndObject();
  /*!
   * \brief Write key value pair in the object.
   * \param key the key of the object.
   * \param value the value of to be written.
   * \tparam ValueType The value type to be written.
   */
  template<typename ValueType>
  inline void WriteObjectKeyValue(const std::string &key,
                                  const ValueType &value);
  /*!
   * \brief Write seperator of array, before writing next element.
   * User can proceed to call writer->Write to write next item
   */
  inline void WriteArraySeperator();
  /*!
   * \brief Write value into array.
   * \param value The value of to be written.
   * \tparam ValueType The value type to be written.
   */
  template<typename ValueType>
  inline void WriteArrayItem(const ValueType &value);
  /*!
   * \brief Write value to json.
   * \param value any STL or json readable that can be written.
   * \tparam ValueType the data type to be write.
   */
  template<typename ValueType>
  inline void Write(const ValueType &value);

 private:
  /*! \brief Output stream */
  std::ostream *os_;
  /*!
   * \brief record how many element processed in
   *  current array/object scope.
   */
  std::vector<size_t> scope_counter_;
  /*! \brief Record whether current is a multiline scope */
  std::vector<bool> scope_multi_line_;
  /*!
   * \brief Write seperating space and newlines
   */
  inline void WriteSeperator();
};

/*!
 * \brief Helper class to read JSON into a class or struct object.
 * \code
 *  struct Param {
 *    std::string name;
 *    int value;
 *    // define load function from JSON
 *    inline void Load(dmlc::JSONReader *reader) {
 *      dmlc::JSONStructReadHelper helper;
 *      helper.DeclareField("name", &name);
 *      helper.DeclareField("value", &value);
 *      helper.ReadAllFields(reader);
 *    }
 *  };
 * \endcode
 */
class JSONObjectReadHelper {
 public:
  /*!
   * \brief Declare field of type T
   * \param key the key of the of field.
   * \param addr address of the data type.
   * \tparam T the data type to be read, must be STL composition of JSON serializable.
   */
  template<typename T>
  inline void DeclareField(const std::string &key, T *addr) {
    DeclareFieldInternal(key, addr, false);
  }
  /*!
   * \brief Declare optional field of type T
   * \param key the key of the of field.
   * \param addr address of the data type.
   * \tparam T the data type to be read, must be STL composition of JSON serializable.
   */
  template<typename T>
  inline void DeclareOptionalField(const std::string &key, T *addr) {
    DeclareFieldInternal(key, addr, true);
  }
  /*!
   * \brief Read in all the declared fields.
   * \param reader the JSONReader to read the json.
   */
  inline void ReadAllFields(JSONReader *reader);

 private:
  /*!
   * \brief Internal function to declare field.
   * \param key the key of the of field.
   * \param addr address of the data type.
   * \param optional if set to true, no error will be reported if the key is not presented.
   * \tparam T the data type to be read, must be STL composition of JSON serializable.
   */
  template<typename T>
  inline void DeclareFieldInternal(const std::string &key, T *addr, bool optional);
  /*!
   * \brief The internal reader function.
   * \param reader The reader to read.
   * \param addr The memory address to read.
   */
  template<typename T>
  inline static void ReaderFunction(JSONReader *reader, void *addr);
  /*! \brief callback type to reader function */
  typedef void (*ReadFunction)(JSONReader *reader, void *addr);
  /*! \brief internal data entry */
  struct Entry {
    /*! \brief the reader function */
    ReadFunction func;
    /*! \brief the address to read */
    void *addr;
    /*! \brief whether it is optional */
    bool optional;
  };
  /*! \brief the internal map of reader callbacks */
  std::map<std::string, Entry> map_;
};

#define DMLC_JSON_ENABLE_ANY_VAR_DEF(KeyName)                  \
  static DMLC_ATTRIBUTE_UNUSED ::dmlc::json::AnyJSONManager&   \
  __make_AnyJSONType ## _ ## KeyName ## __

/*!
 * \def DMLC_JSON_ENABLE_ANY
 * \brief Macro to enable save/load JSON of dmlc:: whose actual type is Type.
 * Any type will be saved as json array [KeyName, content]
 *
 * \param Type The type to be registered.
 * \param KeyName The Type key assigned to the type, must be same during load.
 */
#define DMLC_JSON_ENABLE_ANY(Type, KeyName)                             \
  DMLC_STR_CONCAT(DMLC_JSON_ENABLE_ANY_VAR_DEF(KeyName), __COUNTER__) = \
    ::dmlc::json::AnyJSONManager::Global()->EnableType<Type>(#KeyName) \

//! \cond Doxygen_Suppress
namespace json {

/*!
 * \brief generic serialization handler
 * \tparam T the type to be serialized
 */
template<typename T>
struct Handler;

template<typename ValueType>
struct NumericHandler {
  inline static void Write(JSONWriter *writer, const ValueType &value) {
    writer->WriteNumber<ValueType>(value);
  }
  inline static void Read(JSONReader *reader, ValueType *value) {
    reader->ReadNumber<ValueType>(value);
  }
};

template<typename ContainerType>
struct ArrayHandler {
  inline static void Write(JSONWriter *writer, const ContainerType &array) {
    typedef typename ContainerType::value_type ElemType;
    writer->BeginArray(array.size() > 10 || !dmlc::is_pod<ElemType>::value);
    for (typename ContainerType::const_iterator it = array.begin();
         it != array.end(); ++it) {
      writer->WriteArrayItem(*it);
    }
    writer->EndArray();
  }
  inline static void Read(JSONReader *reader, ContainerType *array) {
    typedef typename ContainerType::value_type ElemType;
    array->clear();
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      ElemType value;
      Handler<ElemType>::Read(reader, &value);
      array->insert(array->end(), value);
    }
  }
};

template<typename ContainerType>
struct MapHandler{
  inline static void Write(JSONWriter *writer, const ContainerType &map) {
    writer->BeginObject(map.size() > 1);
    for (typename ContainerType::const_iterator it = map.begin(); it != map.end(); ++it) {
      writer->WriteObjectKeyValue(it->first, it->second);
    }
    writer->EndObject();
  }
  inline static void Read(JSONReader *reader, ContainerType *map) {
    typedef typename ContainerType::mapped_type ElemType;
    map->clear();
    reader->BeginObject();
    std::string key;
    while (reader->NextObjectItem(&key)) {
      ElemType value;
      reader->Read(&value);
      (*map)[key] = value;
    }
  }
};

template<typename T>
struct CommonJSONSerializer {
  inline static void Write(JSONWriter *writer, const T &value) {
    value.Save(writer);
  }
  inline static void Read(JSONReader *reader, T *value) {
    value->Load(reader);
  }
};

template<>
struct Handler<std::string> {
  inline static void Write(JSONWriter *writer, const std::string &value) {
    writer->WriteString(value);
  }
  inline static void Read(JSONReader *reader, std::string *str) {
    reader->ReadString(str);
  }
};

template<typename T>
struct Handler<std::vector<T> > : public ArrayHandler<std::vector<T> > {
};

template<typename K, typename V>
struct Handler<std::pair<K, V> > {
  inline static void Write(JSONWriter *writer, const std::pair<K, V> &kv) {
    writer->BeginArray();
    writer->WriteArrayItem(kv.first);
    writer->WriteArrayItem(kv.second);
    writer->EndArray();
  }
  inline static void Read(JSONReader *reader, std::pair<K, V> *kv) {
    reader->BeginArray();
    CHECK(reader->NextArrayItem())
        << "Expect array of length 2";
    Handler<K>::Read(reader, &(kv->first));
    CHECK(reader->NextArrayItem())
        << "Expect array of length 2";
    Handler<V>::Read(reader, &(kv->second));
    CHECK(!reader->NextArrayItem())
        << "Expect array of length 2";
  }
};

template<typename T>
struct Handler<std::list<T> > : public ArrayHandler<std::list<T> > {
};

template<typename V>
struct Handler<std::map<std::string, V> > : public MapHandler<std::map<std::string, V> > {
};

#if DMLC_USE_CXX11
template<typename V>
struct Handler<std::unordered_map<std::string, V> >
    : public MapHandler<std::unordered_map<std::string, V> > {
};
#endif  // DMLC_USE_CXX11

template<typename T>
struct Handler {
  inline static void Write(JSONWriter *writer, const T &data) {
    typedef typename dmlc::IfThenElseType<dmlc::is_arithmetic<T>::value,
                                          NumericHandler<T>,
                                          CommonJSONSerializer<T> >::Type THandler;
    THandler::Write(writer, data);
  }
  inline static void Read(JSONReader *reader, T *data) {
    typedef typename dmlc::IfThenElseType<dmlc::is_arithmetic<T>::value,
                                          NumericHandler<T>,
                                          CommonJSONSerializer<T> >::Type THandler;
    THandler::Read(reader, data);
  }
};

#if DMLC_STRICT_CXX11
// Manager to store json serialization strategy.
class AnyJSONManager {
 public:
  template<typename T>
  inline AnyJSONManager& EnableType(const std::string& type_name) {  // NOLINT(*)
    std::type_index tp = std::type_index(typeid(T));
    if (type_name_.count(tp) != 0) {
      CHECK(type_name_.at(tp) == type_name)
          << "Type has already been registered as another typename " << type_name_.at(tp);
      return *this;
    }
    CHECK(type_map_.count(type_name) == 0)
        << "Type name " << type_name << " already registered in registry";
    Entry e;
    e.read = ReadAny<T>;
    e.write = WriteAny<T>;
    type_name_[tp] = type_name;
    type_map_[type_name] = e;
    return *this;
  }
  // return global singleton
  inline static AnyJSONManager* Global() {
    static AnyJSONManager inst;
    return &inst;
  }

 private:
  AnyJSONManager() {}

  template<typename T>
  inline static void WriteAny(JSONWriter *writer, const any &data) {
    writer->Write(dmlc::get<T>(data));
  }
  template<typename T>
  inline static void ReadAny(JSONReader *reader, any* data) {
    T temp;
    reader->Read(&temp);
    *data = std::move(temp);
  }
  // data entry to store vtable for any type
  struct Entry {
    void (*read)(JSONReader* reader, any *data);
    void (*write)(JSONWriter* reader, const any& data);
  };

  template<typename T>
  friend struct Handler;

  std::unordered_map<std::type_index, std::string> type_name_;
  std::unordered_map<std::string, Entry> type_map_;
};

template<>
struct Handler<any> {
  inline static void Write(JSONWriter *writer, const any &data) {
    std::unordered_map<std::type_index, std::string>&
        nmap = AnyJSONManager::Global()->type_name_;
    std::type_index id = std::type_index(data.type());
    auto it = nmap.find(id);
    CHECK(it != nmap.end() && it->first == id)
        << "Type " << id.name() << " has not been registered via DMLC_JSON_ENABLE_ANY";
    std::string type_name = it->second;
    AnyJSONManager::Entry e = AnyJSONManager::Global()->type_map_.at(type_name);
    writer->BeginArray(false);
    writer->WriteArrayItem(type_name);
    writer->WriteArraySeperator();
    e.write(writer, data);
    writer->EndArray();
  }
  inline static void Read(JSONReader *reader, any *data) {
    std::string type_name;
    reader->BeginArray();
    CHECK(reader->NextArrayItem()) << "invalid any json format";
    Handler<std::string>::Read(reader, &type_name);
    std::unordered_map<std::string, AnyJSONManager::Entry>&
        tmap = AnyJSONManager::Global()->type_map_;
    auto it = tmap.find(type_name);
    CHECK(it != tmap.end() && it->first == type_name)
        << "Typename " << type_name << " has not been registered via DMLC_JSON_ENABLE_ANY";
    AnyJSONManager::Entry e = it->second;
    CHECK(reader->NextArrayItem()) << "invalid any json format";
    e.read(reader, data);
    CHECK(!reader->NextArrayItem()) << "invalid any json format";
  }
};
#endif  // DMLC_STRICT_CXX11

}  // namespace json

// implementations of JSONReader/Writer
inline int JSONReader::NextNonSpace() {
  int ch;
  do {
    ch = is_->get();
    if (ch == '\n') ++line_count_n_;
    if (ch == '\r') ++line_count_r_;
  } while (isspace(ch));
  return ch;
}

inline int JSONReader::PeekNextNonSpace() {
  int ch;
  while (true) {
    ch = is_->peek();
    if (ch == '\n') ++line_count_n_;
    if (ch == '\r') ++line_count_r_;
    if (!isspace(ch)) break;
    is_->get();
  }
  return ch;
}

inline void JSONReader::ReadString(std::string *out_str) {
  int ch = NextNonSpace();
  CHECK_EQ(ch, '\"')
      << "Error at" << line_info()
      << ", Expect \'\"\' but get \'" << static_cast<char>(ch) << '\'';
  std::ostringstream os;
  while (true) {
    ch = is_->get();
    if (ch == '\\') {
      char sch = static_cast<char>(is_->get());
      switch (sch) {
        case 'r': os << "\r"; break;
        case 'n': os << "\n"; break;
        case '\\': os << "\\"; break;
        case '\t': os << "\t"; break;
        case '\"': os << "\""; break;
        default: LOG(FATAL) << "unknown string escape \\" << sch;
      }
    } else {
      if (ch == '\"') break;
      os << static_cast<char>(ch);
    }
    if (ch == EOF || ch == '\r' || ch == '\n') {
      LOG(FATAL)
          << "Error at" << line_info()
          << ", Expect \'\"\' but reach end of line ";
    }
  }
  *out_str = os.str();
}

template<typename ValueType>
inline void JSONReader::ReadNumber(ValueType *out_value) {
  *is_ >> *out_value;
  CHECK(!is_->fail())
      << "Error at" << line_info()
      << ", Expect number";
}

inline void JSONReader::BeginObject() {
  int ch = NextNonSpace();
  CHECK_EQ(ch, '{')
      << "Error at" << line_info()
      << ", Expect \'{\' but get \'" << static_cast<char>(ch) << '\'';
  scope_counter_.push_back(0);
}

inline void JSONReader::BeginArray() {
  int ch = NextNonSpace();
  CHECK_EQ(ch, '[')
      << "Error at" << line_info()
      << ", Expect \'{\' but get \'" << static_cast<char>(ch) << '\'';
  scope_counter_.push_back(0);
}

inline bool JSONReader::NextObjectItem(std::string *out_key) {
  bool next = true;
  if (scope_counter_.back() != 0) {
    int ch = NextNonSpace();
    if (ch == EOF) {
      next = false;
    } else if (ch == '}') {
      next = false;
    } else {
      CHECK_EQ(ch, ',')
          << "Error at" << line_info()
          << ", JSON object expect \'}\' or \',\' \'" << static_cast<char>(ch) << '\'';
    }
  } else {
    int ch = PeekNextNonSpace();
    if (ch == '}') {
      is_->get();
      next = false;
    }
  }
  if (!next) {
    scope_counter_.pop_back();
    return false;
  } else {
    scope_counter_.back() += 1;
    ReadString(out_key);
    int ch = NextNonSpace();
    CHECK_EQ(ch, ':')
        << "Error at" << line_info()
        << ", Expect \':\' but get \'" << static_cast<char>(ch) << '\'';
    return true;
  }
}

inline bool JSONReader::NextArrayItem() {
  bool next = true;
  if (scope_counter_.back() != 0) {
    int ch = NextNonSpace();
    if (ch == EOF) {
      next = false;
    } else if (ch == ']') {
      next = false;
    } else {
      CHECK_EQ(ch, ',')
          << "Error at" << line_info()
          << ", JSON array expect \']\' or \',\'. Get \'" << static_cast<char>(ch) << "\' instead";
    }
  } else {
    int ch = PeekNextNonSpace();
    if (ch == ']') {
      is_->get();
      next = false;
    }
  }
  if (!next) {
    scope_counter_.pop_back();
    return false;
  } else {
    scope_counter_.back() += 1;
    return true;
  }
}

template<typename ValueType>
inline void JSONReader::Read(ValueType *out_value) {
  json::Handler<ValueType>::Read(this, out_value);
}

inline void JSONWriter::WriteNoEscape(const std::string &s) {
  *os_ << '\"' << s << '\"';
}

inline void JSONWriter::WriteString(const std::string &s) {
  std::ostream &os = *os_;
  os << '\"';
  for (size_t i = 0; i < s.length(); ++i) {
    char ch = s[i];
    switch (ch) {
      case '\r': os << "\\r"; break;
      case '\n': os << "\\n"; break;
      case '\\': os << "\\\\"; break;
      case '\t': os << "\\t"; break;
      case '\"': os << "\\\""; break;
      default: os << ch;
    }
  }
  os << '\"';
}

template<typename ValueType>
inline void JSONWriter::WriteNumber(const ValueType &v) {
  *os_ << v;
}

inline void JSONWriter::BeginArray(bool multi_line) {
  *os_ << '[';
  scope_multi_line_.push_back(multi_line);
  scope_counter_.push_back(0);
}

inline void JSONWriter::EndArray() {
  CHECK_NE(scope_multi_line_.size(), 0);
  CHECK_NE(scope_counter_.size(), 0);
  bool newline = scope_multi_line_.back();
  size_t nelem = scope_counter_.back();
  scope_multi_line_.pop_back();
  scope_counter_.pop_back();
  if (newline && nelem != 0) WriteSeperator();
  *os_ << ']';
}

inline void JSONWriter::BeginObject(bool multi_line) {
  *os_ << "{";
  scope_multi_line_.push_back(multi_line);
  scope_counter_.push_back(0);
}

inline void JSONWriter::EndObject() {
  CHECK_NE(scope_multi_line_.size(), 0);
  CHECK_NE(scope_counter_.size(), 0);
  bool newline = scope_multi_line_.back();
  size_t nelem = scope_counter_.back();
  scope_multi_line_.pop_back();
  scope_counter_.pop_back();
  if (newline && nelem != 0) WriteSeperator();
  *os_ << '}';
}

template<typename ValueType>
inline void JSONWriter::WriteObjectKeyValue(const std::string &key,
                                            const ValueType &value) {
  std::ostream &os = *os_;
  if (scope_counter_.back() == 0) {
    WriteSeperator();
    os << '\"' << key << "\": ";
  } else {
    os << ", ";
    WriteSeperator();
    os << '\"' << key << "\": ";
  }
  scope_counter_.back() += 1;
  json::Handler<ValueType>::Write(this, value);
}

inline void JSONWriter::WriteArraySeperator() {
  std::ostream &os = *os_;
  if (scope_counter_.back() != 0) {
    os << ", ";
  }
  scope_counter_.back() += 1;
  WriteSeperator();
}

template<typename ValueType>
inline void JSONWriter::WriteArrayItem(const ValueType &value) {
  this->WriteArraySeperator();
  json::Handler<ValueType>::Write(this, value);
}

template<typename ValueType>
inline void JSONWriter::Write(const ValueType &value) {
  size_t nscope = scope_multi_line_.size();
  json::Handler<ValueType>::Write(this, value);
  CHECK_EQ(nscope, scope_multi_line_.size())
      << "Uneven scope, did you call EndArray/EndObject after each BeginObject/Array?";
}

inline void JSONWriter::WriteSeperator() {
  if (scope_multi_line_.size() == 0 || scope_multi_line_.back()) {
    *os_ << '\n' << std::string(scope_multi_line_.size() * 2, ' ');
  }
}

inline void JSONObjectReadHelper::ReadAllFields(JSONReader *reader) {
  reader->BeginObject();
  std::map<std::string, int> visited;
  std::string key;
  while (reader->NextObjectItem(&key)) {
    if (map_.count(key) != 0) {
      Entry e = map_[key];
      (*e.func)(reader, e.addr);
      visited[key] = 0;
    } else {
      std::ostringstream os;
      os << "JSONReader: Unknown field " << key << ", candidates are: \n";
      for (std::map<std::string, Entry>::iterator
               it = map_.begin(); it != map_.end(); ++it) {
        os << '\"' <<it->first << "\"\n";
      }
      LOG(FATAL) << os.str();
    }
  }
  if (visited.size() != map_.size()) {
    for (std::map<std::string, Entry>::iterator
             it = map_.begin(); it != map_.end(); ++it) {
      if (it->second.optional) continue;
      CHECK_NE(visited.count(it->first), 0)
          << "JSONReader: Missing field \"" << it->first << "\"\n At "
          << reader->line_info();
    }
  }
}

template<typename T>
inline void JSONObjectReadHelper::ReaderFunction(JSONReader *reader, void *addr) {
  json::Handler<T>::Read(reader, static_cast<T*>(addr));
}

template<typename T>
inline void JSONObjectReadHelper::
DeclareFieldInternal(const std::string &key, T *addr, bool optional) {
  CHECK_EQ(map_.count(key), 0)
      << "Adding duplicate field " << key;
  Entry e;
  e.func = ReaderFunction<T>;
  e.addr = static_cast<void*>(addr);
  e.optional = optional;
  map_[key] = e;
}

//! \endcond
}  // namespace dmlc
#endif  // DMLC_JSON_H_
