/*!
 * Copyright 2018 by Contributors
 * \file parameter.h
 * \brief macro for using C++11 enum class as DMLC parameter
 * \author Hyunsu Philip Cho
 */

#ifndef XGBOOST_PARAMETER_H_
#define XGBOOST_PARAMETER_H_

#include <dmlc/parameter.h>
#include <xgboost/base.h>
#include <string>
#include <type_traits>

/*!
 * \brief Specialization of FieldEntry for enum class (backed by int)
 *
 * Use this macro to use C++11 enum class as DMLC parameters
 *
 * Usage:
 *
 * \code{.cpp}
 *
 *   // enum class must inherit from int type
 *   enum class Foo : int {
 *     kBar = 0, kFrog = 1, kCat = 2, kDog = 3
 *   };
 *
 *   // This line is needed to prevent compilation error
 *   DECLARE_FIELD_ENUM_CLASS(Foo);
 *
 *   // Now define DMLC parameter as usual;
 *   //   enum classes can now be members.
 *   struct MyParam : dmlc::Parameter<MyParam> {
 *     Foo foo;
 *     DMLC_DECLARE_PARAMETER(MyParam) {
 *       DMLC_DECLARE_FIELD(foo)
 *         .set_default(Foo::kBar)
 *         .add_enum("bar", Foo::kBar)
 *         .add_enum("frog", Foo::kFrog)
 *         .add_enum("cat", Foo::kCat)
 *         .add_enum("dog", Foo::kDog);
 *     }
 *   };
 *
 *   DMLC_REGISTER_PARAMETER(MyParam);
 * \endcode
 */
#define DECLARE_FIELD_ENUM_CLASS(EnumClass) \
namespace dmlc {  \
namespace parameter {  \
template <>  \
class FieldEntry<EnumClass> : public FieldEntry<int> {  \
 public:  \
  FieldEntry<EnumClass>() {  \
    static_assert(  \
      std::is_same<int, typename std::underlying_type<EnumClass>::type>::value,  \
      "enum class must be backed by int");  \
    is_enum_ = true;  \
  }  \
  using Super = FieldEntry<int>;  \
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
  inline void Init(const std::string &key, void *head, EnumClass& ref) {  /* NOLINT */  \
    Super::Init(key, head, *reinterpret_cast<int*>(&ref));  \
  }  \
};  \
}  /* namespace parameter */  \
}  /* namespace dmlc */

namespace xgboost {
template <typename Type>
struct XGBoostParameter : public dmlc::Parameter<Type> {
 protected:
  bool initialised_ {false};

 public:
  template <typename Container>
  Args UpdateAllowUnknown(Container const& kwargs) {
    if (initialised_) {
      return dmlc::Parameter<Type>::UpdateAllowUnknown(kwargs);
    } else {
      auto unknown = dmlc::Parameter<Type>::InitAllowUnknown(kwargs);
      initialised_ = true;
      return unknown;
    }
  }
  bool GetInitialised() const { return static_cast<bool>(this->initialised_); }
};
}  // namespace xgboost

#endif  // XGBOOST_PARAMETER_H_
