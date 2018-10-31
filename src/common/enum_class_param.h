/*!
 * Copyright 2015-2018 by Contributors
 * \file enum_class_param.h
 * \brief macro for using C++11 enum class as DMLC parameter
 * \author Hyunsu Philip Cho
 */

#ifndef XGBOOST_COMMON_ENUM_CLASS_PARAM_H_
#define XGBOOST_COMMON_ENUM_CLASS_PARAM_H_

#include <dmlc/parameter.h>
#include <string>
#include <type_traits>

// specialization of FieldEntry for enum class (backed by int)
#define DECLARE_FIELD_ENUM_CLASS(EnumClass) \
template <>  \
class dmlc::parameter::FieldEntry< EnumClass >  \
  : public dmlc::parameter::FieldEntry<int> {  \
 public:  \
  FieldEntry<EnumClass>() {  \
    static_assert(  \
      std::is_same<int, typename std::underlying_type<EnumClass>::type>::value,  \
      "enum class must be backed by int");  \
    is_enum_ = true;  \
  }  \
  typedef FieldEntry<int> Super;  \
  void Set(void *head, const std::string &value) const override {  \
    Super::Set(head, value);  \
  }  \
  inline FieldEntry<EnumClass>& add_enum(const std::string &key, EnumClass value) {  \
    Super::add_enum(key, static_cast<int>(value));  \
    return *this;  \
  }  \
  inline FieldEntry<EnumClass>& set_default(const EnumClass& default_value) {  \
    default_value_ = static_cast<int>(default_value);  \
    has_default_ = true;  \
    return *this;  \
  }  \
  inline void Init(const std::string &key, void *head, EnumClass& ref) {  \
    Super::Init(key, head, *reinterpret_cast<int*>(&ref));  \
  }  \
};

#endif  // XGBOOST_COMMON_ENUM_CLASS_PARAM_H_
