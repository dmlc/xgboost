/**
 * Copyright 2022-2025, XGBoost contributors
 */
#pragma once
#include <cuda_runtime.h>

#include <memory>  // for unique_ptr

#include "common.h"
namespace xgboost::curt {
class StreamView;

class Event {
  std::unique_ptr<cudaEvent_t, void (*)(cudaEvent_t *)> event_;

 public:
  explicit Event(bool disable_timing = true)
      : event_{[disable_timing] {
                 auto e = new cudaEvent_t;
                 dh::safe_cuda(cudaEventCreateWithFlags(
                     e, disable_timing ? cudaEventDisableTiming : cudaEventDefault));
                 return e;
               }(),
               [](cudaEvent_t *e) {
                 if (e) {
                   dh::safe_cuda(cudaEventDestroy(*e));
                   delete e;
                 }
               }} {}

  inline void Record(StreamView stream);  // NOLINT
  // Define swap-based ctor to make sure an event is always valid.
  Event(Event &&e) : Event() { std::swap(this->event_, e.event_); }
  Event &operator=(Event &&e) {
    std::swap(this->event_, e.event_);
    return *this;
  }

  operator cudaEvent_t() const { return *event_; }                // NOLINT
  cudaEvent_t const *data() const { return this->event_.get(); }  // NOLINT
  void Sync() { dh::safe_cuda(cudaEventSynchronize(*this->data())); }
};

class StreamView {
  cudaStream_t stream_{nullptr};

 public:
  explicit StreamView(cudaStream_t s) : stream_{s} {}
  void Wait(Event const &e) {
#if defined(__CUDACC_VER_MAJOR__)
#if __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 0
    // CUDA == 11.0
    dh::safe_cuda(cudaStreamWaitEvent(stream_, cudaEvent_t{e}, 0));
#else
    // CUDA > 11.0
    dh::safe_cuda(cudaStreamWaitEvent(stream_, cudaEvent_t{e}, cudaEventWaitDefault));
#endif  // __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 0:
#else   // clang
    dh::safe_cuda(cudaStreamWaitEvent(stream_, cudaEvent_t{e}, cudaEventWaitDefault));
#endif  //  defined(__CUDACC_VER_MAJOR__)
  }
  operator cudaStream_t() const {  // NOLINT
    return stream_;
  }
  cudaError_t Sync(bool error = true) {
    if (error) {
      dh::safe_cuda(cudaStreamSynchronize(stream_));
      return cudaSuccess;
    }
    return cudaStreamSynchronize(stream_);
  }
};

inline void Event::Record(StreamView stream) {  // NOLINT
  dh::safe_cuda(cudaEventRecord(*event_, cudaStream_t{stream}));
}

// Changing this has effect on prediction return, where we need to pass the pointer to
// third-party libraries like cuPy
inline StreamView DefaultStream() { return StreamView{cudaStreamPerThread}; }

class Stream {
  cudaStream_t stream_;

 public:
  Stream() { dh::safe_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~Stream() { dh::safe_cuda(cudaStreamDestroy(stream_)); }

  [[nodiscard]] StreamView View() const { return StreamView{stream_}; }
  [[nodiscard]] cudaStream_t Handle() const { return stream_; }

  void Sync() { this->View().Sync(); }
  void Wait(Event const &e) { this->View().Wait(e); }
};
}  // namespace xgboost::curt
