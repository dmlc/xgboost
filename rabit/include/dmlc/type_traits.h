/*!
 *  Copyright (c) 2015 by Contributors
 * \file type_traits.h
 * \brief type traits information header
 */
#ifndef DMLC_TYPE_TRAITS_H_
#define DMLC_TYPE_TRAITS_H_

#include "./base.h"
#if DMLC_USE_CXX11
#include <type_traits>
#endif
#include <string>

namespace dmlc {
/*!
 * \brief whether a type is pod type
 * \tparam T the type to query
 */
template<typename T>
struct is_pod {
#if DMLC_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_pod<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = false;
#endif
};


/*!
 * \brief whether a type is integer type
 * \tparam T the type to query
 */
template<typename T>
struct is_integral {
#if DMLC_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_integral<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = false;
#endif
};

/*!
 * \brief whether a type is floating point type
 * \tparam T the type to query
 */
template<typename T>
struct is_floating_point {
#if DMLC_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_floating_point<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = false;
#endif
};

/*!
 * \brief whether a type is arithemetic type
 * \tparam T the type to query
 */
template<typename T>
struct is_arithmetic {
#if DMLC_USE_CXX11
  /*! \brief the value of the traits */
  static const bool value = std::is_arithmetic<T>::value;
#else
  /*! \brief the value of the traits */
  static const bool value = (dmlc::is_integral<T>::value ||
                             dmlc::is_floating_point<T>::value);
#endif
};

/*!
 * \brief the string representation of type name
 * \tparam T the type to query
 * \return a const string of typename.
 */
template<typename T>
inline const char* type_name() {
  return "";
}

/*!
 * \brief whether a type have save/load function
 * \tparam T the type to query
 */
template<typename T>
struct has_saveload {
  /*! \brief the value of the traits */
  static const bool value = false;
};

/*!
 * \brief template to select type based on condition
 * For example, IfThenElseType<true, int, float>::Type will give int
 * \tparam cond the condition
 * \tparam Then the typename to be returned if cond is true
 * \tparam The typename to be returned if cond is false
*/
template<bool cond, typename Then, typename Else>
struct IfThenElseType;

/*! \brief macro to quickly declare traits information */
#define DMLC_DECLARE_TRAITS(Trait, Type, Value)       \
  template<>                                          \
  struct Trait<Type> {                                \
    static const bool value = Value;                  \
  }

/*! \brief macro to quickly declare traits information */
#define DMLC_DECLARE_TYPE_NAME(Type, Name)            \
  template<>                                          \
  inline const char* type_name<Type>() {              \
    return Name;                                      \
  }

//! \cond Doxygen_Suppress
// declare special traits when C++11 is not available
#if DMLC_USE_CXX11 == 0
DMLC_DECLARE_TRAITS(is_pod, char, true);
DMLC_DECLARE_TRAITS(is_pod, int8_t, true);
DMLC_DECLARE_TRAITS(is_pod, int16_t, true);
DMLC_DECLARE_TRAITS(is_pod, int32_t, true);
DMLC_DECLARE_TRAITS(is_pod, int64_t, true);
DMLC_DECLARE_TRAITS(is_pod, uint8_t, true);
DMLC_DECLARE_TRAITS(is_pod, uint16_t, true);
DMLC_DECLARE_TRAITS(is_pod, uint32_t, true);
DMLC_DECLARE_TRAITS(is_pod, uint64_t, true);
DMLC_DECLARE_TRAITS(is_pod, float, true);
DMLC_DECLARE_TRAITS(is_pod, double, true);

DMLC_DECLARE_TRAITS(is_integral, char, true);
DMLC_DECLARE_TRAITS(is_integral, int8_t, true);
DMLC_DECLARE_TRAITS(is_integral, int16_t, true);
DMLC_DECLARE_TRAITS(is_integral, int32_t, true);
DMLC_DECLARE_TRAITS(is_integral, int64_t, true);
DMLC_DECLARE_TRAITS(is_integral, uint8_t, true);
DMLC_DECLARE_TRAITS(is_integral, uint16_t, true);
DMLC_DECLARE_TRAITS(is_integral, uint32_t, true);
DMLC_DECLARE_TRAITS(is_integral, uint64_t, true);

DMLC_DECLARE_TRAITS(is_floating_point, float, true);
DMLC_DECLARE_TRAITS(is_floating_point, double, true);

#endif

DMLC_DECLARE_TYPE_NAME(float, "float");
DMLC_DECLARE_TYPE_NAME(double, "double");
DMLC_DECLARE_TYPE_NAME(int, "int");
DMLC_DECLARE_TYPE_NAME(uint32_t, "int (non-negative)");
DMLC_DECLARE_TYPE_NAME(uint64_t, "long (non-negative)");
DMLC_DECLARE_TYPE_NAME(std::string, "string");
DMLC_DECLARE_TYPE_NAME(bool, "boolean");

template<typename Then, typename Else>
struct IfThenElseType<true, Then, Else> {
  typedef Then Type;
};

template<typename Then, typename Else>
struct IfThenElseType<false, Then, Else> {
  typedef Else Type;
};
//! \endcond
}  // namespace dmlc
#endif  // DMLC_TYPE_TRAITS_H_
