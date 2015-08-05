/*!
 * Copyright 2014 by Contributors
 * \file thread_buffer.h
 * \brief  multi-thread buffer, iterator, can be used to create parallel pipeline
 * \author Tianqi Chen
 */
#ifndef XGBOOST_UTILS_THREAD_BUFFER_H_
#define XGBOOST_UTILS_THREAD_BUFFER_H_

#include <vector>
#include <cstring>
#include <cstdlib>
#include "./utils.h"
// threading util could not run on solaris
#ifndef XGBOOST_STRICT_CXX98_
#include "./thread.h"
#endif

namespace xgboost {
namespace utils {
#if !defined(XGBOOST_STRICT_CXX98_)
/*!
 * \brief buffered loading iterator that uses multithread
 * this template method will assume the following paramters
 * \tparam Elem elememt type to be buffered
 * \tparam ElemFactory factory type to implement in order to use thread buffer
 */
template<typename Elem, typename ElemFactory>
class ThreadBuffer {
 public:
  /*!\brief constructor */
  ThreadBuffer(void) {
    this->init_end = false;
    this->buf_size = 30;
  }
  ~ThreadBuffer(void) {
    if (init_end) this->Destroy();
  }
  /*!\brief set parameter, will also pass the parameter to factory */
  inline void SetParam(const char *name, const char *val) {
    using namespace std;
    if (!strcmp( name, "buffer_size")) buf_size = atoi(val);
    factory.SetParam(name, val);
  }
  /*!
   * \brief initalize the buffered iterator
   * \param param a initialize parameter that will pass to factory, ignore it if not necessary
   * \return false if the initlization can't be done, e.g. buffer file hasn't been created
   */
  inline bool Init(void) {
    if (!factory.Init()) return false;
    for (int i = 0; i < buf_size; ++i) {
      bufA.push_back(factory.Create());
      bufB.push_back(factory.Create());
    }
    this->init_end = true;
    this->StartLoader();
    return true;
  }
  /*!\brief place the iterator before first value */
  inline void BeforeFirst(void) {
    // wait till last loader end
    loading_end.Wait();
    // critcal zone
    current_buf = 1;
    factory.BeforeFirst();
    // reset terminate limit
    endA = endB = buf_size;
    // wake up loader for first part
    loading_need.Post();
    // wait til first part is loaded
    loading_end.Wait();
    // set current buf to right value
    current_buf = 0;
    // wake loader for next part
    data_loaded = false;
    loading_need.Post();
    // set buffer value
    buf_index = 0;
  }
  /*! \brief destroy the buffer iterator, will deallocate the buffer */
  inline void Destroy(void) {
    // wait until the signal is consumed
    this->destroy_signal = true;
    loading_need.Post();
    loader_thread.Join();
    loading_need.Destroy();
    loading_end.Destroy();
    for (size_t i = 0; i < bufA.size(); ++i) {
      factory.FreeSpace(bufA[i]);
    }
    for (size_t i = 0; i < bufB.size(); ++i) {
      factory.FreeSpace(bufB[i]);
    }
    bufA.clear(); bufB.clear();
    factory.Destroy();
    this->init_end = false;
  }
  /*!
   * \brief get the next element needed in buffer
   * \param elem element to store into
   * \return whether reaches end of data
   */
  inline bool Next(Elem &elem) { // NOLINT(*)
    // end of buffer try to switch
    if (buf_index == buf_size) {
      this->SwitchBuffer();
      buf_index = 0;
    }
    if (buf_index >= (current_buf ? endA : endB)) {
      return false;
    }
    std::vector<Elem> &buf = current_buf ? bufA : bufB;
    elem = buf[buf_index];
    ++buf_index;
    return true;
  }
  /*!
   * \brief get the factory object
   */
  inline ElemFactory &get_factory(void) {
    return factory;
  }
  inline const ElemFactory &get_factory(void) const {
    return factory;
  }
  // size of buffer
  int  buf_size;

 private:
  // factory object used to load configures
  ElemFactory factory;
  // index in current buffer
  int buf_index;
  // indicate which one is current buffer
  int current_buf;
  // max limit of visit, also marks termination
  int endA, endB;
  // double buffer, one is accessed by loader
  // the other is accessed by consumer
  // buffer of the data
  std::vector<Elem> bufA, bufB;
  // initialization end
  bool init_end;
  // singal whether the data is loaded
  bool data_loaded;
  // signal to kill the thread
  bool destroy_signal;
  // thread object
  Thread loader_thread;
  // signal of the buffer
  Semaphore loading_end, loading_need;
  /*!
   * \brief slave thread
   * this implementation is like producer-consumer style
   */
  inline void RunLoader(void) {
    while (!destroy_signal) {
      // sleep until loading is needed
      loading_need.Wait();
      std::vector<Elem> &buf = current_buf ? bufB : bufA;
      int i;
      for (i = 0; i < buf_size ; ++i) {
        if (!factory.LoadNext(buf[i])) {
          int &end = current_buf ? endB : endA;
          end = i;  // marks the termination
          break;
        }
      }
      // signal that loading is done
      data_loaded = true;
      loading_end.Post();
    }
  }
  /*!\brief entry point of loader thread */
  inline static XGBOOST_THREAD_PREFIX LoaderEntry(void *pthread) {
    static_cast< ThreadBuffer<Elem, ElemFactory>* >(pthread)->RunLoader();
    return NULL;
  }
  /*!\brief start loader thread */
  inline void StartLoader(void) {
    destroy_signal = false;
    // set param
    current_buf = 1;
    loading_need.Init(1);
    loading_end .Init(0);
    // reset terminate limit
    endA = endB = buf_size;
    loader_thread.Start(LoaderEntry, this);
    // wait until first part of data is loaded
    loading_end.Wait();
    // set current buf to right value
    current_buf = 0;
    // wake loader for next part
    data_loaded = false;
    loading_need.Post();
    buf_index = 0;
  }
  /*!\brief switch double buffer */
  inline void SwitchBuffer(void) {
    loading_end.Wait();
    // loader shall be sleep now, critcal zone!
    current_buf = !current_buf;
    // wake up loader
    data_loaded = false;
    loading_need.Post();
  }
};
#else
// a dummy single threaded ThreadBuffer
// use this to resolve R's solaris compatibility for now
template<typename Elem, typename ElemFactory>
class ThreadBuffer {
 public:
  ThreadBuffer() : init_end_(false) {}
  ~ThreadBuffer() {
    if (init_end_) {
      factory_.FreeSpace(data_);
      factory_.Destroy();
    }
  }
  inline void SetParam(const char *name, const char *val) {
  }
  inline bool Init(void) {
    if (!factory_.Init()) return false;
    data_ = factory_.Create();
    return (init_end_ = true);
  }
  inline void BeforeFirst(void) {
    factory_.BeforeFirst();
  }
  inline bool Next(Elem &elem) { // NOLINT(*)
    if (factory_.LoadNext(data_)) {
      elem = data_; return true;
    } else {
      return false;
    }
  }
  inline ElemFactory &get_factory() {
    return factory_;
  }
  inline const ElemFactory &get_factory() const {
    return factory_;
  }

 private:
  // initialized
  bool init_end_;
  // current data
  Elem data_;
  // factory object used to load configures
  ElemFactory factory_;
};
#endif  // !defined(XGBOOST_STRICT_CXX98_)
}  // namespace utils
}  // namespace xgboost
#endif  // XGBOOST_UTILS_THREAD_BUFFER_H_
