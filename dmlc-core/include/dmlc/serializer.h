/*!
 *  Copyright (c) 2015 by Contributors
 * \file serializer.h
 * \brief serializer template class that helps serialization.
 *  This file do not need to be directly used by most user.
 */
#ifndef DMLC_SERIALIZER_H_
#define DMLC_SERIALIZER_H_

#include <vector>
#include <string>
#include <map>
#include <set>
#include <list>
#include <deque>
#include <utility>

#include "./base.h"
#include "./io.h"
#include "./logging.h"
#include "./type_traits.h"

#if DMLC_USE_CXX11
#include <unordered_map>
#include <unordered_set>
#endif

namespace dmlc {
/*! \brief internal namespace for serializers */
namespace serializer {
/*!
 * \brief generic serialization handler
 * \tparam T the type to be serialized
 */
template<typename T>
struct Handler;

//! \cond Doxygen_Suppress
/*!
 * \brief Serializer that redirect calls by condition
 * \tparam cond the condition
 * \tparam Then the serializer used for then condition
 * \tparam Else the serializer used for else condition
 * \tparam Return the type of data the serializer handles
 */
template<bool cond, typename Then, typename Else, typename Return>
struct IfThenElse;

template<typename Then, typename Else, typename T>
struct IfThenElse<true, Then, Else, T> {
  inline static void Write(Stream *strm, const T &data) {
    Then::Write(strm, data);
  }
  inline static bool Read(Stream *strm, T *data) {
    return Then::Read(strm, data);
  }
};
template<typename Then, typename Else, typename T>
struct IfThenElse<false, Then, Else, T> {
  inline static void Write(Stream *strm, const T &data) {
    Else::Write(strm, data);
  }
  inline static bool Read(Stream *strm, T *data) {
    return Else::Read(strm, data);
  }
};

/*! \brief Serializer for POD(plain-old-data) data */
template<typename T>
struct PODHandler {
  inline static void Write(Stream *strm, const T &data) {
    strm->Write(&data, sizeof(T));
  }
  inline static bool Read(Stream *strm, T *dptr) {
    return strm->Read((void*)dptr, sizeof(T)) == sizeof(T);  // NOLINT(*)
  }
};

// serializer for class that have save/load function
template<typename T>
struct SaveLoadClassHandler {
  inline static void Write(Stream *strm, const T &data) {
    data.Save(strm);
  }
  inline static bool Read(Stream *strm, T *data) {
    return data->Load(strm);
  }
};

/*!
 * \brief dummy class for undefined serialization.
 *   This is used to generate error message when user tries to
 *   serialize something that is not supported.
 * \tparam T the type to be serialized
 */
template<typename T>
struct UndefinedSerializerFor {
};

/*!
 * \brief Serializer handler for std::vector<T> where T is POD type.
 * \tparam T element type
 */
template<typename T>
struct PODVectorHandler {
  inline static void Write(Stream *strm, const std::vector<T> &vec) {
    uint64_t sz = static_cast<uint64_t>(vec.size());
    strm->Write(&sz, sizeof(sz));
    if (sz != 0) {
      strm->Write(&vec[0], sizeof(T) * vec.size());
    }
  }
  inline static bool Read(Stream *strm, std::vector<T> *out_vec) {
    uint64_t sz;
    if (strm->Read(&sz, sizeof(sz)) != sizeof(sz)) return false;
    size_t size = static_cast<size_t>(sz);
    out_vec->resize(size);
    if (sz != 0) {
      size_t nbytes = sizeof(T) * size;
      return strm->Read(&(*out_vec)[0], nbytes) == nbytes;
    }
    return true;
  }
};

/*!
 * \brief Serializer handler for std::vector<T> where T can be composed type
 * \tparam T element type
 */
template<typename T>
struct ComposeVectorHandler {
  inline static void Write(Stream *strm, const std::vector<T> &vec) {
    uint64_t sz = static_cast<uint64_t>(vec.size());
    strm->Write(&sz, sizeof(sz));
    for (size_t i = 0; i < vec.size(); ++i) {
      Handler<T>::Write(strm, vec[i]);
    }
  }
  inline static bool Read(Stream *strm, std::vector<T> *out_vec) {
    uint64_t sz;
    if (strm->Read(&sz, sizeof(sz)) != sizeof(sz)) return false;
    size_t size = static_cast<size_t>(sz);
    out_vec->resize(size);
    for (size_t i = 0; i < size; ++i) {
      if (!Handler<T>::Read(strm, &(*out_vec)[i])) return false;
    }
    return true;
  }
};

/*!
 * \brief Serializer handler for std::basic_string<T> where T is POD type.
 * \tparam T element type
 */
template<typename T>
struct PODStringHandler {
  inline static void Write(Stream *strm, const std::basic_string<T> &vec) {
    uint64_t sz = static_cast<uint64_t>(vec.length());
    strm->Write(&sz, sizeof(sz));
    if (sz != 0) {
      strm->Write(&vec[0], sizeof(T) * vec.length());
    }
  }
  inline static bool Read(Stream *strm, std::basic_string<T> *out_vec) {
    uint64_t sz;
    if (strm->Read(&sz, sizeof(sz)) != sizeof(sz)) return false;
    size_t size = static_cast<size_t>(sz);
    out_vec->resize(size);
    if (sz != 0) {
      size_t nbytes = sizeof(T) * size;
      return strm->Read(&(*out_vec)[0], nbytes) == nbytes;
    }
    return true;
  }
};

/*! \brief Serializer for std::pair */
template<typename TA, typename TB>
struct PairHandler {
  inline static void Write(Stream *strm, const std::pair<TA, TB> &data) {
    Handler<TA>::Write(strm, data.first);
    Handler<TB>::Write(strm, data.second);
  }
  inline static bool Read(Stream *strm, std::pair<TA, TB> *data) {
    return Handler<TA>::Read(strm, &(data->first)) &&
        Handler<TB>::Read(strm, &(data->second));
  }
};

// set type handler that can handle most collection type case
template<typename ContainerType>
struct CollectionHandler {
  inline static void Write(Stream *strm, const ContainerType &data) {
    typedef typename ContainerType::value_type ElemType;
    // dump data to vector
    std::vector<ElemType> vdata(data.begin(), data.end());
    // serialize the vector
    Handler<std::vector<ElemType> >::Write(strm, vdata);
  }
  inline static bool Read(Stream *strm, ContainerType *data) {
    typedef typename ContainerType::value_type ElemType;
    std::vector<ElemType> vdata;
    if (!Handler<std::vector<ElemType> >::Read(strm, &vdata)) return false;
    data->clear();
    data->insert(vdata.begin(), vdata.end());
    return true;
  }
};


// handler that can handle most list type case
// this type insert function takes additional iterator
template<typename ListType>
struct ListHandler {
  inline static void Write(Stream *strm, const ListType &data) {
    typedef typename ListType::value_type ElemType;
    // dump data to vector
    std::vector<ElemType> vdata(data.begin(), data.end());
    // serialize the vector
    Handler<std::vector<ElemType> >::Write(strm, vdata);
  }
  inline static bool Read(Stream *strm, ListType *data) {
    typedef typename ListType::value_type ElemType;
    std::vector<ElemType> vdata;
    if (!Handler<std::vector<ElemType> >::Read(strm, &vdata)) return false;
    data->clear();
    data->insert(data->begin(), vdata.begin(), vdata.end());
    return true;
  }
};

//! \endcond

/*!
 * \brief generic serialization handler for type T
 *
 *  User can define specialization of this class to support
 *  composite serialization of their own class.
 *
 * \tparam T the type to be serialized
 */
template<typename T>
struct Handler {
  /*!
   * \brief write data to stream
   * \param strm the stream we write the data.
   * \param data the data obeject to be serialized
   */
  inline static void Write(Stream *strm, const T &data) {
    IfThenElse<dmlc::is_pod<T>::value,
               PODHandler<T>,
               IfThenElse<dmlc::has_saveload<T>::value,
                          SaveLoadClassHandler<T>,
                          UndefinedSerializerFor<T>, T>,
               T>
        ::Write(strm, data);
  }
  /*!
   * \brief read data to stream
   * \param strm the stream to read the data.
   * \param data the pointer to the data obeject to read
   * \return whether the read is successful
   */
  inline static bool Read(Stream *strm, T *data) {
    return IfThenElse<dmlc::is_pod<T>::value,
                      PODHandler<T>,
                      IfThenElse<dmlc::has_saveload<T>::value,
                                 SaveLoadClassHandler<T>,
                                 UndefinedSerializerFor<T>, T>,
                      T>
    ::Read(strm, data);
  }
};

//! \cond Doxygen_Suppress
template<typename T>
struct Handler<std::vector<T> > {
  inline static void Write(Stream *strm, const std::vector<T> &data) {
    IfThenElse<dmlc::is_pod<T>::value,
               PODVectorHandler<T>,
               ComposeVectorHandler<T>, std::vector<T> >
    ::Write(strm, data);
  }
  inline static bool Read(Stream *strm, std::vector<T> *data) {
    return IfThenElse<dmlc::is_pod<T>::value,
                      PODVectorHandler<T>,
                      ComposeVectorHandler<T>,
                      std::vector<T> >
    ::Read(strm, data);
  }
};

template<typename T>
struct Handler<std::basic_string<T> > {
  inline static void Write(Stream *strm, const std::basic_string<T> &data) {
    IfThenElse<dmlc::is_pod<T>::value,
               PODStringHandler<T>,
               UndefinedSerializerFor<T>,
               std::basic_string<T> >
    ::Write(strm, data);
  }
  inline static bool Read(Stream *strm, std::basic_string<T> *data) {
    return IfThenElse<dmlc::is_pod<T>::value,
                      PODStringHandler<T>,
                      UndefinedSerializerFor<T>,
                      std::basic_string<T> >
    ::Read(strm, data);
  }
};

template<typename TA, typename TB>
struct Handler<std::pair<TA, TB> > {
  inline static void Write(Stream *strm, const std::pair<TA, TB> &data) {
    IfThenElse<dmlc::is_pod<TA>::value && dmlc::is_pod<TB>::value,
               PODHandler<std::pair<TA, TB> >,
               PairHandler<TA, TB>,
               std::pair<TA, TB> >
    ::Write(strm, data);
  }
  inline static bool Read(Stream *strm, std::pair<TA, TB> *data) {
    return IfThenElse<dmlc::is_pod<TA>::value && dmlc::is_pod<TB>::value,
                      PODHandler<std::pair<TA, TB> >,
                      PairHandler<TA, TB>,
                      std::pair<TA, TB> >
    ::Read(strm, data);
  }
};

template<typename K, typename V>
struct Handler<std::map<K, V> >
    : public CollectionHandler<std::map<K, V> > {
};

template<typename K, typename V>
struct Handler<std::multimap<K, V> >
    : public CollectionHandler<std::multimap<K, V> > {
};

template<typename T>
struct Handler<std::set<T> >
    : public CollectionHandler<std::set<T> > {
};

template<typename T>
struct Handler<std::multiset<T> >
    : public CollectionHandler<std::multiset<T> > {
};

template<typename T>
struct Handler<std::list<T> >
    : public ListHandler<std::list<T> > {
};

template<typename T>
struct Handler<std::deque<T> >
    : public ListHandler<std::deque<T> > {
};

#if DMLC_USE_CXX11
template<typename K, typename V>
struct Handler<std::unordered_map<K, V> >
    : public CollectionHandler<std::unordered_map<K, V> > {
};

template<typename K, typename V>
struct Handler<std::unordered_multimap<K, V> >
    : public CollectionHandler<std::unordered_multimap<K, V> > {
};

template<typename T>
struct Handler<std::unordered_set<T> >
    : public CollectionHandler<std::unordered_set<T> > {
};

template<typename T>
struct Handler<std::unordered_multiset<T> >
    : public CollectionHandler<std::unordered_multiset<T> > {
};
#endif
//! \endcond
}  // namespace serializer
}  // namespace dmlc
#endif  // DMLC_SERIALIZER_H_
