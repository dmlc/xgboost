/*!
 * Copyright (c) 2016 by Contributors
 * \file any.h
 * \brief Container to hold any data type.
 */
#ifndef DMLC_ANY_H_
#define DMLC_ANY_H_

// This code need c++11 to compile
#include <typeinfo>
#include <type_traits>
#include <utility>
#include <algorithm>

#include "./base.h"
#include "./logging.h"

namespace dmlc {
// forward declare any;
class any;

/*!
 * Get a  reference to content stored in the any as type T.
 * This will cause an error if
 * T does not match the type stored.
 * This function is not part of std::any standard.
 *
 * \param src The source source any container.
 * \return The reference of content
 * \tparam T The type of the value to be fetched.
 */
template<typename T>
inline T& get(any& src);  // NOLINT(*)

/*!
 * Get the const reference content stored in the any as type T.
 * This will cause an error if
 * T does not match the type stored.
 * This function is not part of std::any standard.
 *
 * \param src The source source any container.
 * \return The reference of content
 * \tparam T The type of the value to be fetched.
 */
template<typename T>
inline const T& get(const any& src);

/*!
 * \brief An any class that is compatible to std::any in c++17.
 *
 * \code
 *   dmlc::any a = std::string("mydear"), b = 1;
 *   // get reference out and add it
 *   dmlc::get<int>(b) += 1;
 *   // a is now string
 *   LOG(INFO) << dmlc::get<std::string>(a);
 *   // a is now 2, the string stored will be properly destructed
 *   a = std::move(b);
 *   LOG(INFO) << dmlc::get<int>(a);
 * \endcode
 * \sa get
 */
class any {
 public:
  /*! \brief default constructor */
  inline any() = default;
  /*!
   * \brief move constructor from another any
   * \param other The other any to be moved
   */
  inline any(any&& other);  // NOLINT(*)
  /*!
   * \brief copy constructor
   * \param other The other any to be copied
   */
  inline any(const any& other);  // NOLINT(*)
  /*!
   * \brief constructor from any types
   * \param other The other types to be constructed into any.
   * \tparam T The value type of other.
   */
  template<typename T>
  inline any(T&& other);  // NOLINT(*)
  /*! \brief destructor */
  inline ~any();
  /*!
   * \brief assign operator from other
   * \param other The other any to be copy or moved.
   * \return self
   */
  inline any& operator=(any&& other);
  /*!
   * \brief assign operator from other
   * \param other The other any to be copy or moved.
   * \return self
   */
  inline any& operator=(const any& other);
  /*!
   * \brief assign operator from any type.
   * \param other The other any to be copy or moved.
   * \tparam T The value type of other.
   * \return self
   */
  template<typename T>
  inline any& operator=(T&& other);
  /*!
   * \return whether the container is empty.
   */
  inline bool empty() const;
  /*!
   * \return clear the content of container
   */
  inline void clear();
  /*!
   * swap current content with other
   * \param other The other data to be swapped.
   */
  inline void swap(any& other); // NOLINT(*)
  /*!
   * \return The type_info about the stored type.
   */
  inline const std::type_info& type() const;

 private:
  //! \cond Doxygen_Suppress
  // declare of helper class
  template<typename T>
  class TypeOnHeap;
  template<typename T>
  class TypeOnStack;
  template<typename T>
  class TypeInfo;
  // size of stack space, it takes 32 bytes for one any type.
  static const size_t kStack = sizeof(void*) * 3;
  static const size_t kAlign = sizeof(void*);
  // container use dynamic storage only when space runs lager
  union Data {
    // stack space
    std::aligned_storage<kStack, kAlign>::type stack;
    // pointer to heap space
    void* pheap;
  };
  // type specific information
  struct Type {
    // destructor function
    void (*destroy)(Data* data);
    // copy constructor
    void (*create_from_data)(Data* dst, const Data& src);
    // the type info function
    const std::type_info* ptype_info;
  };
  // constant to check if data can be stored on heap.
  template<typename T>
  struct data_on_stack {
    static const bool value = alignof(T) <= kAlign && sizeof(T) <= kStack;
  };
  // declare friend with
  template<typename T>
  friend T& get(any& src);  // NOLINT(*)
  template<typename T>
  friend const T& get(const any& src);
  // internal construct function
  inline void construct(any&& other);
  // internal construct function
  inline void construct(const any& other);
  // internal function to check if type is correct.
  template<typename T>
  inline void check_type() const;
  // internal type specific information
  const Type* type_{nullptr};
  // internal data
  Data data_;
};

template<typename T>
inline any::any(T&& other) {
  typedef typename std::decay<T>::type DT;
  if (std::is_same<DT, any>::value) {
    this->construct(std::forward<T>(other));
  } else {
    static_assert(std::is_copy_constructible<DT>::value,
                  "Any can only hold value that is copy constructable");
    type_ = TypeInfo<DT>::get_type();
    if (data_on_stack<DT>::value) {
      new (&(data_.stack)) DT(std::forward<T>(other));
    } else {
      data_.pheap = new DT(std::forward<T>(other));
    }
  }
}

inline any::any(any&& other) {
  this->construct(std::move(other));
}

inline any::any(const any& other) {
  this->construct(other);
}

inline void any::construct(any&& other) {
  type_ = other.type_;
  data_ = other.data_;
  other.type_ = nullptr;
}

inline void any::construct(const any& other) {
  type_ = other.type_;
  if (type_ != nullptr) {
    type_->create_from_data(&data_, other.data_);
  }
}

inline any::~any() {
  this->clear();
}

inline any& any::operator=(any&& other) {
  any(std::move(other)).swap(*this);
  return *this;
}

inline any& any::operator=(const any& other) {
  any(other).swap(*this);
  return *this;
}

template<typename T>
inline any& any::operator=(T&& other) {
  any(std::forward<T>(other)).swap(*this);
  return *this;
}

inline void any::swap(any& other) { // NOLINT(*)
  std::swap(type_, other.type_);
  std::swap(data_, other.data_);
}

inline void any::clear() {
  if (type_ != nullptr) {
    if (type_->destroy != nullptr) {
      type_->destroy(&data_);
    }
    type_ = nullptr;
  }
}

inline bool any::empty() const {
  return type_ == nullptr;
}

inline const std::type_info& any::type() const {
  if (type_ != nullptr) {
    return *(type_->ptype_info);
  } else {
    return typeid(void);
  }
}

template<typename T>
inline void any::check_type() const {
  CHECK(type_ != nullptr)
      << "The any container is empty"
      << " requested=" << typeid(T).name();
  CHECK(type_->ptype_info == &typeid(T))
      << "The stored type mismatch"
      << " stored=" << type_->ptype_info->name()
      << " requested=" << typeid(T).name();
}

template<typename T>
inline const T& get(const any& src) {
  src.check_type<T>();
  return *any::TypeInfo<T>::get_ptr(&(src.data_));
}

template<typename T>
inline T& get(any& src) { // NOLINT(*)
  src.check_type<T>();
  return *any::TypeInfo<T>::get_ptr(&(src.data_));
}

template<typename T>
class any::TypeOnHeap {
 public:
  inline static T* get_ptr(any::Data* data) {
    return static_cast<T*>(data->pheap);
  }
  inline static const T* get_ptr(const any::Data* data) {
    return static_cast<const T*>(data->pheap);
  }
  inline static void create_from_data(any::Data* dst, const any::Data& data) {
    dst->pheap = new T(*get_ptr(&data));
  }
  inline static void destroy(Data* data) {
    delete static_cast<T*>(data->pheap);
  }
};

template<typename T>
class any::TypeOnStack {
 public:
  inline static T* get_ptr(any::Data* data) {
    return reinterpret_cast<T*>(&(data->stack));
  }
  inline static const T* get_ptr(const any::Data* data) {
    return reinterpret_cast<const T*>(&(data->stack));
  }
  inline static void create_from_data(any::Data* dst, const any::Data& data) {
    new (&(dst->stack)) T(*get_ptr(&data));
  }
  inline static void destroy(Data* data) {
    T* dptr = reinterpret_cast<T*>(&(data->stack));
    dptr->~T();
  }
};

template<typename T>
class any::TypeInfo
    : public std::conditional<any::data_on_stack<T>::value,
                              any::TypeOnStack<T>,
                              any::TypeOnHeap<T> >::type {
 public:
  inline static const Type* get_type() {
    static TypeInfo<T> tp;
    return &(tp.type_);
  }

 private:
  // local type
  Type type_;
  // constructor
  TypeInfo() {
    if (std::is_pod<T>::value) {
      type_.destroy = nullptr;
    } else {
      type_.destroy = TypeInfo<T>::destroy;
    }
    type_.create_from_data = TypeInfo<T>::create_from_data;
    type_.ptype_info = &typeid(T);
  }
};
//! \endcond

}  // namespace dmlc

#endif  // DMLC_ANY_H_
