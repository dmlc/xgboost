/*!
 * Copyright (c) 2015 by Contributors
 * \file parameter.h
 * \brief Provide lightweight util to do parameter setup and checking.
 */
#ifndef DMLC_PARAMETER_H_
#define DMLC_PARAMETER_H_

#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <limits>
#include <map>
#include <set>
#include <typeinfo>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>
#include "./base.h"
#include "./json.h"
#include "./logging.h"
#include "./type_traits.h"

namespace dmlc {
// this file is backward compatible with non-c++11
/*! \brief Error throwed by parameter checking */
struct ParamError : public dmlc::Error {
  /*!
   * \brief constructor
   * \param msg error message
   */
  explicit ParamError(const std::string &msg)
      : dmlc::Error(msg) {}
};

/*!
 * \brief Get environment variable with default.
 * \param key the name of environment variable.
 * \param default_value the default value of environment vriable.
 * \return The value received
 */
template<typename ValueType>
inline ValueType GetEnv(const char *key,
                        ValueType default_value);

/*! \brief internal namespace for parameter manangement */
namespace parameter {
// forward declare ParamManager
class ParamManager;
// forward declare FieldAccessEntry
class FieldAccessEntry;
// forward declare FieldEntry
template<typename DType>
class FieldEntry;
// forward declare ParamManagerSingleton
template<typename PType>
struct ParamManagerSingleton;

/*! \brief option in parameter initialization */
enum ParamInitOption {
  /*! \brief allow unknown parameters */
  kAllowUnknown,
  /*! \brief need to match exact parameters */
  kAllMatch
};
}  // namespace parameter
/*!
 * \brief Information about a parameter field in string representations.
 */
struct ParamFieldInfo {
  /*! \brief name of the field */
  std::string name;
  /*! \brief type of the field in string format */
  std::string type;
  /*!
   * \brief detailed type information string
   *  This include the default value, enum constran and typename.
   */
  std::string type_info_str;
  /*! \brief detailed description of the type */
  std::string description;
};

/*!
 * \brief Parameter is the base type every parameter struct should inheritate from
 * The following code is a complete example to setup parameters.
 * \code
 *   struct Param : public dmlc::Parameter<Param> {
 *     float learning_rate;
 *     int num_hidden;
 *     std::string name;
 *     // declare parameters in header file
 *     DMLC_DECLARE_PARAMETER(Param) {
 *       DMLC_DECLARE_FIELD(num_hidden).set_range(0, 1000);
 *       DMLC_DECLARE_FIELD(learning_rate).set_default(0.01f);
 *       DMLC_DECLARE_FIELD(name).set_default("hello");
 *     }
 *   };
 *   // register it in cc file
 *   DMLC_REGISTER_PARAMETER(Param);
 * \endcode
 *
 *  After that, the Param struct will get all the functions defined in Parameter.
 * \tparam PType the type of parameter struct
 *
 * \sa DMLC_DECLARE_FIELD, DMLC_REGISTER_PARAMETER, DMLC_DECLARE_PARAMETER
 */
template<typename PType>
struct Parameter {
 public:
  /*!
   * \brief initialize the parameter by keyword arguments.
   *  This function will initialize the parameter struct, check consistency
   *  and throw error if something wrong happens.
   *
   * \param kwargs map of keyword arguments, or vector of pairs
   * \parma option The option on initialization.
   * \tparam Container container type
   * \throw ParamError when something go wrong.
   */
  template<typename Container>
  inline void Init(const Container &kwargs,
                   parameter::ParamInitOption option = parameter::kAllowUnknown) {
    PType::__MANAGER__()->RunInit(static_cast<PType*>(this),
                                  kwargs.begin(), kwargs.end(),
                                  NULL,
                                  option == parameter::kAllowUnknown);
  }
  /*!
   * \brief initialize the parameter by keyword arguments.
   *  This is same as Init, but allow unknown arguments.
   *
   * \param kwargs map of keyword arguments, or vector of pairs
   * \tparam Container container type
   * \throw ParamError when something go wrong.
   * \return vector of pairs of unknown arguments.
   */
  template<typename Container>
  inline std::vector<std::pair<std::string, std::string> >
  InitAllowUnknown(const Container &kwargs) {
    std::vector<std::pair<std::string, std::string> > unknown;
    PType::__MANAGER__()->RunInit(static_cast<PType*>(this),
                                  kwargs.begin(), kwargs.end(),
                                  &unknown, true);
    return unknown;
  }
  /*!
   * \brief Return a dictionary representation of the parameters
   * \return A dictionary that maps key -> value
   */
  inline std::map<std::string, std::string> __DICT__() const {
    std::vector<std::pair<std::string, std::string> > vec
        = PType::__MANAGER__()->GetDict(this->head());
    return std::map<std::string, std::string>(vec.begin(), vec.end());
  }
  /*!
   * \brief Write the parameters in JSON format.
   * \param writer JSONWriter used for writing.
   */
  inline void Save(dmlc::JSONWriter *writer) const {
    writer->Write(this->__DICT__());
  }
  /*!
   * \brief Load the parameters from JSON.
   * \param reader JSONReader used for loading.
   * \throw ParamError when something go wrong.
   */
  inline void Load(dmlc::JSONReader *reader) {
    std::map<std::string, std::string> kwargs;
    reader->Read(&kwargs);
    this->Init(kwargs);
  }
  /*!
   * \brief Get the fields of the parameters.
   * \return List of ParamFieldInfo of each field.
   */
  inline static std::vector<ParamFieldInfo> __FIELDS__() {
    return PType::__MANAGER__()->GetFieldInfo();
  }
  /*!
   * \brief Print docstring of the parameter
   * \return the printed docstring
   */
  inline static std::string __DOC__() {
    std::ostringstream os;
    PType::__MANAGER__()->PrintDocString(os);
    return os.str();
  }

 protected:
  /*!
   * \brief internal function to allow declare of a parameter memember
   * \param manager the parameter manager
   * \param key the key name of the parameter
   * \param ref the reference to the parameter in the struct.
   */
  template<typename DType>
  inline parameter::FieldEntry<DType>& DECLARE(
      parameter::ParamManagerSingleton<PType> *manager,
      const std::string &key, DType &ref) { // NOLINT(*)
    parameter::FieldEntry<DType> *e =
        new parameter::FieldEntry<DType>();
    e->Init(key, this->head(), ref);
    manager->manager.AddEntry(key, e);
    return *e;
  }

 private:
  /*! \return Get head pointer of child structure */
  inline PType *head() const {
    return static_cast<PType*>(const_cast<Parameter<PType>*>(this));
  }
};

//! \cond Doxygen_Suppress
/*!
 * \brief macro used to declare parameter
 *
 * Example:
 * \code
 *   struct Param : public dmlc::Parameter<Param> {
 *     // declare parameters in header file
 *     DMLC_DECLARE_PARAMETER(Param) {
 *        // details of declarations
 *     }
 *   };
 * \endcode
 *
 * This macro need to be put in a source file so that registeration only happens once.
 * Refer to example code in Parameter for details
 *
 * \param PType the name of parameter struct.
 * \sa Parameter
 */
#define DMLC_DECLARE_PARAMETER(PType)                                   \
  static ::dmlc::parameter::ParamManager *__MANAGER__();                \
  inline void __DECLARE__(::dmlc::parameter::ParamManagerSingleton<PType> *manager) \

/*!
 * \brief macro to declare fields
 * \param FieldName the name of the field.
 */
#define DMLC_DECLARE_FIELD(FieldName)  this->DECLARE(manager, #FieldName, FieldName)

/*!
 * \brief macro to declare alias of a fields
 * \param FieldName the name of the field.
 * \param AliasName the name of the alias, must be declared after the field is declared.
 */
#define DMLC_DECLARE_ALIAS(FieldName, AliasName)  manager->manager.AddAlias(#FieldName, #AliasName)

/*!
 * \brief Macro used to register parameter.
 *
 * This macro need to be put in a source file so that registeration only happens once.
 * Refer to example code in Parameter for details
 * \param PType the type of parameter struct.
 * \sa Parameter
 */
#define DMLC_REGISTER_PARAMETER(PType)                                  \
  ::dmlc::parameter::ParamManager *PType::__MANAGER__() {               \
    static ::dmlc::parameter::ParamManagerSingleton<PType> inst(#PType); \
    return &inst.manager;                                               \
  }                                                                     \
  static DMLC_ATTRIBUTE_UNUSED ::dmlc::parameter::ParamManager&         \
  __make__ ## PType ## ParamManager__ =                                 \
      (*PType::__MANAGER__())                                           \

//! \endcond
/*!
 * \brief internal namespace for parameter manangement
 * There is no need to use it directly in normal case
 */
namespace parameter {
/*!
 * \brief FieldAccessEntry interface to help manage the parameters
 *  Each entry can be used to access one parameter in the Parameter struct.
 *
 *  This is an internal interface used that is used to manage parameters
 */
class FieldAccessEntry {
 public:
  FieldAccessEntry()
      : has_default_(false) {}
  /*! \brief destructor */
  virtual ~FieldAccessEntry() {}
  /*!
   * \brief set the default value.
   * \param head the pointer to the head of the struct
   * \throw error if no default is presented
   */
  virtual void SetDefault(void *head) const = 0;
  /*!
   * \brief set the parameter by string value
   * \param head the pointer to the head of the struct
   * \param value the value to be set
   */
  virtual void Set(void *head, const std::string &value) const = 0;
  // check if value is OK
  virtual void Check(void *head) const {}
  /*!
   * \brief get the string representation of value.
   * \param head the pointer to the head of the struct
   */
  virtual std::string GetStringValue(void *head) const = 0;
  /*!
   * \brief Get field information
   * \return the corresponding field information
   */
  virtual ParamFieldInfo GetFieldInfo() const = 0;

 protected:
  /*! \brief whether this parameter have default value */
  bool has_default_;
  /*! \brief positional index of parameter in struct */
  size_t index_;
  /*! \brief parameter key name */
  std::string key_;
  /*! \brief parameter type */
  std::string type_;
  /*! \brief description of the parameter */
  std::string description_;
  /*!
   * \brief print string representation of default value
   * \parma os the stream to print the docstring to.
   */
  virtual void PrintDefaultValueString(std::ostream &os) const = 0;  // NOLINT(*)
  // allow ParamManager to modify self
  friend class ParamManager;
};

/*!
 * \brief manager class to handle parameter setting for each type
 *  An manager will be created for each parameter types.
 */
class ParamManager {
 public:
  /*! \brief destructor */
  ~ParamManager() {
    for (size_t i = 0; i < entry_.size(); ++i) {
      delete entry_[i];
    }
  }
  /*!
   * \brief find the access entry by parameter key
   * \param key the key of the parameter.
   * \return pointer to FieldAccessEntry, NULL if nothing is found.
   */
  inline FieldAccessEntry *Find(const std::string &key) const {
    std::map<std::string, FieldAccessEntry*>::const_iterator it =
        entry_map_.find(key);
    if (it == entry_map_.end()) return NULL;
    return it->second;
  }
  /*!
   * \brief set parameter by keyword arguments.
   * \param head head to the parameter field.
   * \param begin begin iterator of original kwargs
   * \param end end iterator of original kwargs
   * \param unknown_args optional, used to hold unknown arguments
   *          When it is specified, unknown arguments will be stored into here, instead of raise an error
   * \tparam RandomAccessIterator iterator type
   * \throw ParamError when there is unknown argument and unknown_args == NULL, or required argument is missing.
   */
  template<typename RandomAccessIterator>
  inline void RunInit(void *head,
                      RandomAccessIterator begin,
                      RandomAccessIterator end,
                      std::vector<std::pair<std::string, std::string> > *unknown_args,
                      bool allow_unknown) const {
    std::set<FieldAccessEntry*> selected_args;
    for (RandomAccessIterator it = begin; it != end; ++it) {
      FieldAccessEntry *e = Find(it->first);
      if (e != NULL) {
        e->Set(head, it->second);
        e->Check(head);
        selected_args.insert(e);
      } else {
        if (unknown_args != NULL) {
          unknown_args->push_back(*it);
        } else {
          if (!allow_unknown) {
            std::ostringstream os;
            os << "Cannot find argument \'" << it->first << "\', Possible Arguments:\n";
            os << "----------------\n";
            PrintDocString(os);
            throw dmlc::ParamError(os.str());
          }
        }
      }
    }

    for (std::map<std::string, FieldAccessEntry*>::const_iterator it = entry_map_.begin();
         it != entry_map_.end(); ++it) {
      if (selected_args.count(it->second) == 0) {
        it->second->SetDefault(head);
      }
    }
  }
  /*!
   * \brief internal function to add entry to manager,
   *  The manager will take ownership of the entry.
   * \param key the key to the parameters
   * \param e the pointer to the new entry.
   */
  inline void AddEntry(const std::string &key, FieldAccessEntry *e) {
    e->index_ = entry_.size();
    // TODO(bing) better error message
    if (entry_map_.count(key) != 0) {
      LOG(FATAL) << "key " << key << " has already been registered in " << name_;
    }
    entry_.push_back(e);
    entry_map_[key] = e;
  }
  /*!
   * \brief internal function to add entry to manager,
   *  The manager will take ownership of the entry.
   * \param key the key to the parameters
   * \param e the pointer to the new entry.
   */
  inline void AddAlias(const std::string& field, const std::string& alias) {
    if (entry_map_.count(field) == 0) {
      LOG(FATAL) << "key " << field << " has not been registered in " << name_;
    }
    if (entry_map_.count(alias) != 0) {
      LOG(FATAL) << "Alias " << alias << " has already been registered in " << name_;
    }
    entry_map_[alias] = entry_map_[field];
  }
  /*!
   * \brief set the name of parameter manager
   * \param name the name to set
   */
  inline void set_name(const std::string &name) {
    name_ = name;
  }
  /*!
   * \brief get field information of each field.
   * \return field information
   */
  inline std::vector<ParamFieldInfo> GetFieldInfo() const {
    std::vector<ParamFieldInfo> ret(entry_.size());
    for (size_t i = 0; i < entry_.size(); ++i) {
      ret[i] = entry_[i]->GetFieldInfo();
    }
    return ret;
  }
  /*!
   * \brief Print readible docstring to ostream, add newline.
   * \parma os the stream to print the docstring to.
   */
  inline void PrintDocString(std::ostream &os) const {  // NOLINT(*)
    for (size_t i = 0; i < entry_.size(); ++i) {
      ParamFieldInfo info = entry_[i]->GetFieldInfo();
      os << info.name << " : " << info.type_info_str << '\n';
      if (info.description.length() != 0) {
        os << "    " << info.description << '\n';
      }
    }
  }
  /*!
   * \brief Get internal parameters in vector of pairs.
   * \param head the head of the struct.
   * \param skip_default skip the values that equals default value.
   * \return the parameter dictionary.
   */
  inline std::vector<std::pair<std::string, std::string> > GetDict(void * head) const {
    std::vector<std::pair<std::string, std::string> > ret;
    for (std::map<std::string, FieldAccessEntry*>::const_iterator
            it = entry_map_.begin(); it != entry_map_.end(); ++it) {
      ret.push_back(std::make_pair(it->first, it->second->GetStringValue(head)));
    }
    return ret;
  }

 private:
  /*! \brief parameter struct name */
  std::string name_;
  /*! \brief positional list of entries */
  std::vector<FieldAccessEntry*> entry_;
  /*! \brief map of key to entry */
  std::map<std::string, FieldAccessEntry*> entry_map_;
};

//! \cond Doxygen_Suppress

// The following piece of code will be template heavy and less documented
// singleton parameter manager for certain type, used for initialization
template<typename PType>
struct ParamManagerSingleton {
  ParamManager manager;
  explicit ParamManagerSingleton(const std::string &param_name) {
    PType param;
    param.__DECLARE__(this);
    manager.set_name(param_name);
  }
};

// Base class of FieldEntry
// implement set_default
template<typename TEntry, typename DType>
class FieldEntryBase : public FieldAccessEntry {
 public:
  // entry type
  typedef TEntry EntryType;
  // implement set value
  virtual void Set(void *head, const std::string &value) const {
    std::istringstream is(value);
    is >> this->Get(head);
    if (!is.fail()) {
      while (!is.eof()) {
        int ch = is.get();
        if (ch == EOF) {
          is.clear(); break;
        }
        if (!isspace(ch)) {
          is.setstate(std::ios::failbit); break;
        }
      }
    }

    if (is.fail()) {
      std::ostringstream os;
      os << "Invalid Parameter format for " << key_
         << " expect " << type_ << " but value=\'" << value<< '\'';
      throw dmlc::ParamError(os.str());
    }
  }
  virtual std::string GetStringValue(void *head) const {
    std::ostringstream os;
    PrintValue(os, this->Get(head));
    return os.str();
  }
  virtual ParamFieldInfo GetFieldInfo() const {
    ParamFieldInfo info;
    std::ostringstream os;
    info.name = key_;
    info.type = type_;
    os << type_;
    if (has_default_) {
      os << ',' << " optional, default=";
      PrintDefaultValueString(os);
    } else {
      os << ", required";
    }
    info.type_info_str = os.str();
    info.description = description_;
    return info;
  }
  // implement set head to default value
  virtual void SetDefault(void *head) const {
    if (!has_default_) {
      std::ostringstream os;
      os << "Required parameter " << key_
         << " of " << type_ << " is not presented";
      throw dmlc::ParamError(os.str());
    } else {
      this->Get(head) = default_value_;
    }
  }
  // return reference of self as derived type
  inline TEntry &self() {
    return *(static_cast<TEntry*>(this));
  }
  // implement set_default
  inline TEntry &set_default(const DType &default_value) {
    default_value_ = default_value;
    has_default_ = true;
    // return self to allow chaining
    return this->self();
  }
  // implement describe
  inline TEntry &describe(const std::string &description) {
    description_ = description;
    // return self to allow chaining
    return this->self();
  }
  // initialization function
  inline void Init(const std::string &key,
                   void *head, DType &ref) { // NOLINT(*)
    this->key_ = key;
    if (this->type_.length() == 0) {
      this->type_ = dmlc::type_name<DType>();
    }
    this->offset_ = ((char*)&ref) - ((char*)head);  // NOLINT(*)
  }

 protected:
  // print the value
  virtual void PrintValue(std::ostream &os, DType value) const { // NOLINT(*)
    os << value;
  }
  virtual void PrintDefaultValueString(std::ostream &os) const {  // NOLINT(*)
    PrintValue(os, default_value_);
  }
  // get the internal representation of parameter
  // for example if this entry corresponds field param.learning_rate
  // then Get(&param) will return reference to param.learning_rate
  inline DType &Get(void *head) const {
    return *(DType*)((char*)(head) + offset_);  // NOLINT(*)
  }
  // internal offset of the field
  ptrdiff_t offset_;
  // default value of field
  DType default_value_;
};

// parameter base for numeric types that have range
template<typename TEntry, typename DType>
class FieldEntryNumeric
    : public FieldEntryBase<TEntry, DType> {
 public:
  FieldEntryNumeric()
      : has_begin_(false), has_end_(false) {}
  // implement set_range
  virtual TEntry &set_range(DType begin, DType end) {
    begin_ = begin; end_ = end;
    has_begin_ = true; has_end_ = true;
    return this->self();
  }
  // implement set_range
  virtual TEntry &set_lower_bound(DType begin) {
    begin_ = begin; has_begin_ = true;
    return this->self();
  }
  // consistency check for numeric ranges
  virtual void Check(void *head) const {
    FieldEntryBase<TEntry, DType>::Check(head);
    DType v = this->Get(head);
    if (has_begin_ && has_end_) {
      if (v < begin_ || v > end_) {
        std::ostringstream os;
        os << "value " << v << "for Parameter " << this->key_
           << " exceed bound [" << begin_ << ',' << end_ <<']';
        throw dmlc::ParamError(os.str());
      }
    } else if (has_begin_ && v < begin_) {
        std::ostringstream os;
        os << "value " << v << "for Parameter " << this->key_
           << " should be greater equal to " << begin_;
        throw dmlc::ParamError(os.str());
    } else if (has_end_ && v > end_) {
        std::ostringstream os;
        os << "value " << v << "for Parameter " << this->key_
           << " should be smaller equal to " << end_;
        throw dmlc::ParamError(os.str());
    }
  }

 protected:
  // whether it have begin and end range
  bool has_begin_, has_end_;
  // data bound
  DType begin_, end_;
};

/*!
 * \brief FieldEntry defines parsing and checking behavior of DType.
 * This class can be specialized to implement specific behavior of more settings.
 * \tparam DType the data type of the entry.
 */
template<typename DType>
class FieldEntry :
      public IfThenElseType<dmlc::is_arithmetic<DType>::value,
                            FieldEntryNumeric<FieldEntry<DType>, DType>,
                            FieldEntryBase<FieldEntry<DType>, DType> >::Type {
};

// specialize define for int(enum)
template<>
class FieldEntry<int>
    : public FieldEntryNumeric<FieldEntry<int>, int> {
 public:
  // construct
  FieldEntry<int>() : is_enum_(false) {}
  // parent
  typedef FieldEntryNumeric<FieldEntry<int>, int> Parent;
  // override set
  virtual void Set(void *head, const std::string &value) const {
    if (is_enum_) {
      std::map<std::string, int>::const_iterator it = enum_map_.find(value);
      std::ostringstream os;
      if (it == enum_map_.end()) {
        os << "Invalid Input: \'" << value;
        os << "\', valid values are: ";
        PrintEnums(os);
        throw dmlc::ParamError(os.str());
      } else {
        os << it->second;
        Parent::Set(head, os.str());
      }
    } else {
      Parent::Set(head, value);
    }
  }
  virtual ParamFieldInfo GetFieldInfo() const {
    if (is_enum_) {
      ParamFieldInfo info;
      std::ostringstream os;
      info.name = key_;
      info.type = type_;
      PrintEnums(os);
      if (has_default_) {
        os << ',' << "optional, default=";
        PrintDefaultValueString(os);
      } else {
        os << ", required";
      }
      info.type_info_str = os.str();
      info.description = description_;
      return info;
    } else {
      return Parent::GetFieldInfo();
    }
  }
  // add enum
  inline FieldEntry<int> &add_enum(const std::string &key, int value) {
    if ((enum_map_.size() != 0 && enum_map_.count(key) != 0) || \
        enum_back_map_.count(value) != 0) {
      std::ostringstream os;
      os << "Enum " << "(" << key << ": " << value << " exisit!" << ")\n";
      os << "Enums: ";
      for (std::map<std::string, int>::const_iterator it = enum_map_.begin();
           it != enum_map_.end(); ++it) {
        os << "(" << it->first << ": " << it->second << "), ";
      }
      throw dmlc::ParamError(os.str());
    }
    enum_map_[key] = value;
    enum_back_map_[value] = key;
    is_enum_ = true;
    return this->self();
  }

 protected:
  // enum flag
  bool is_enum_;
  // enum map
  std::map<std::string, int> enum_map_;
  // enum map
  std::map<int, std::string> enum_back_map_;
  // override print behavior
  virtual void PrintDefaultValueString(std::ostream &os) const { // NOLINT(*)
    os << '\'';
    PrintValue(os, default_value_);
    os << '\'';
  }
  // override print default
  virtual void PrintValue(std::ostream &os, int value) const {  // NOLINT(*)
    if (is_enum_) {
      CHECK_NE(enum_back_map_.count(value), 0)
          << "Value not found in enum declared";
      os << enum_back_map_.at(value);
    } else {
      os << value;
    }
  }


 private:
  inline void PrintEnums(std::ostream &os) const {  // NOLINT(*)
    os << '{';
    for (std::map<std::string, int>::const_iterator
             it = enum_map_.begin(); it != enum_map_.end(); ++it) {
      if (it != enum_map_.begin()) {
        os << ", ";
      }
      os << "\'" << it->first << '\'';
    }
    os << '}';
  }
};

// specialize define for string
template<>
class FieldEntry<std::string>
    : public FieldEntryBase<FieldEntry<std::string>, std::string> {
 public:
  // parent class
  typedef FieldEntryBase<FieldEntry<std::string>, std::string> Parent;
  // override set
  virtual void Set(void *head, const std::string &value) const {
    this->Get(head) = value;
  }
  // override print default
  virtual void PrintDefaultValueString(std::ostream &os) const {  // NOLINT(*)
    os << '\'' << default_value_ << '\'';
  }
};

// specialize define for bool
template<>
class FieldEntry<bool>
    : public FieldEntryBase<FieldEntry<bool>, bool> {
 public:
  // parent class
  typedef FieldEntryBase<FieldEntry<bool>, bool> Parent;
  // override set
  virtual void Set(void *head, const std::string &value) const {
    std::string lower_case; lower_case.resize(value.length());
    std::transform(value.begin(), value.end(), lower_case.begin(), ::tolower);
    bool &ref = this->Get(head);
    if (lower_case == "true") {
      ref = true;
    } else if (lower_case == "false") {
      ref = false;
    } else if (lower_case == "1") {
      ref = true;
    } else if (lower_case == "0") {
      ref = false;
    } else {
      std::ostringstream os;
      os << "Invalid Parameter format for " << key_
         << " expect " << type_ << " but value=\'" << value<< '\'';
      throw dmlc::ParamError(os.str());
    }
  }

 protected:
  // print default string
  virtual void PrintValue(std::ostream &os, bool value) const {  // NOLINT(*)
    if (value) {
      os << "True";
    } else {
      os << "False";
    }
  }
};

}  // namespace parameter
//! \endcond

// implement GetEnv
template<typename ValueType>
inline ValueType GetEnv(const char *key,
                        ValueType default_value) {
  const char *val = getenv(key);
  if (val == NULL) return default_value;
  ValueType ret;
  parameter::FieldEntry<ValueType> e;
  e.Init(key, &ret, ret);
  e.Set(&ret, val);
  return ret;
}
}  // namespace dmlc
#endif  // DMLC_PARAMETER_H_
