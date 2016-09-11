/*!
 *  Copyright (c) 2015 by Contributors
 * \file thread_local.h
 * \brief Common utility for thread local storage.
 */
#ifndef RABIT_THREAD_LOCAL_H_
#define RABIT_THREAD_LOCAL_H_

#include "../include/dmlc/base.h"

#if DMLC_ENABLE_STD_THREAD
#include <mutex>
#endif

#include <memory>
#include <vector>

namespace rabit {

// macro hanlding for threadlocal variables
#ifdef __GNUC__
  #define MX_TREAD_LOCAL __thread
#elif __STDC_VERSION__ >= 201112L
  #define  MX_TREAD_LOCAL _Thread_local
#elif defined(_MSC_VER)
  #define MX_TREAD_LOCAL __declspec(thread)
#endif

#ifndef MX_TREAD_LOCAL
#message("Warning: Threadlocal is not enabled");
#endif

/*!
 * \brief A threadlocal store to store threadlocal variables.
 *  Will return a thread local singleton of type T
 * \tparam T the type we like to store
 */
template<typename T>
class ThreadLocalStore {
 public:
  /*! \return get a thread local singleton */
  static T* Get() {
    static MX_TREAD_LOCAL T* ptr = nullptr;
    if (ptr == nullptr) {
      ptr = new T();
      Singleton()->RegisterDelete(ptr);
    }
    return ptr;
  }

 private:
  /*! \brief constructor */
  ThreadLocalStore() {}
  /*! \brief destructor */
  ~ThreadLocalStore() {
    for (size_t i = 0; i < data_.size(); ++i) {
      delete data_[i];
    }
  }
  /*! \return singleton of the store */
  static ThreadLocalStore<T> *Singleton() {
    static ThreadLocalStore<T> inst;
    return &inst;
  }
  /*!
   * \brief register str for internal deletion
   * \param str the string pointer
   */
  void RegisterDelete(T *str) {
#if DMLC_ENABLE_STD_THREAD
    std::unique_lock<std::mutex> lock(mutex_);
    data_.push_back(str);
    lock.unlock();
#else
    data_.push_back(str);
#endif
  }

#if DMLC_ENABLE_STD_THREAD
  /*! \brief internal mutex */
  std::mutex mutex_;
#endif
  /*!\brief internal data */
  std::vector<T*> data_;
};
}  // namespace rabit
#endif  // RABIT_THREAD_LOCAL_H_
