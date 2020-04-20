/*!
 * Copyright 2018 XGBoost contributors
 */
#ifndef XGBOOST_COMMON_TRANSFORM_H_
#define XGBOOST_COMMON_TRANSFORM_H_

#include <dmlc/omp.h>
#include <dmlc/common.h>

#include <xgboost/data.h>
#include <utility>
#include <vector>
#include <type_traits>  // enable_if

#include "xgboost/host_device_vector.h"
#include "xgboost/span.h"

#include "common.h"

#if defined (__CUDACC__)
#include "device_helpers.cuh"
#endif  // defined (__CUDACC__)

namespace xgboost {
namespace common {

constexpr size_t kBlockThreads = 256;

namespace detail {

#if defined(__CUDACC__)
template <typename Functor, typename... SpanType>
__global__ void LaunchCUDAKernel(Functor _func, Range _range,
                                 SpanType... _spans) {
  for (auto i : dh::GridStrideRange(*_range.begin(), *_range.end())) {
    _func(i, _spans...);
  }
}
#endif  // defined(__CUDACC__)

}  // namespace detail

/*! \brief Do Transformation on HostDeviceVectors.
 *
 *  \tparam CompiledWithCuda A bool parameter used to distinguish compilation
 *         trajectories, users do not need to use it.
 *
 *  Note: Using Transform is a VERY tricky thing to do. Transform uses template
 *   argument to duplicate itself into two different types, one for CPU,
 *   another for CUDA.  The trick is not without its flaw:
 *
 *     If you use it in a function that can be compiled by both nvcc and host
 *     compiler, the behaviour is un-defined!  Because your function is NOT
 *     duplicated by `CompiledWithCuda`. At link time, cuda compiler resolution
 *     will merge functions with same signature.
 */
template <bool CompiledWithCuda = WITH_CUDA()>
class Transform {
 private:
  template <typename Functor>
  struct Evaluator {
   public:
    Evaluator(Functor func, Range range, int device, bool shard) :
        func_(func), range_{std::move(range)},
        shard_{shard},
        device_{device} {}

    /*!
     * \brief Evaluate the functor with input pointers to HostDeviceVector.
     *
     * \tparam HDV...  HostDeviceVectors type.
     * \param  vectors Pointers to HostDeviceVector.
     */
    template <typename... HDV>
    void Eval(HDV... vectors) const {
      bool on_device = device_ >= 0;

      if (on_device) {
        LaunchCUDA(func_, vectors...);
      } else {
        LaunchCPU(func_, vectors...);
      }
    }

   private:
    // CUDA UnpackHDV
    template <typename T>
    Span<T> UnpackHDVOnDevice(HostDeviceVector<T>* _vec) const {
      auto span = _vec->DeviceSpan();
      return span;
    }
    template <typename T>
    Span<T const> UnpackHDVOnDevice(const HostDeviceVector<T>* _vec) const {
      auto span = _vec->ConstDeviceSpan();
      return span;
    }
    // CPU UnpackHDV
    template <typename T>
    Span<T> UnpackHDV(HostDeviceVector<T>* _vec) const {
      return Span<T> {_vec->HostPointer(),
            static_cast<typename Span<T>::index_type>(_vec->Size())};
    }
    template <typename T>
    Span<T const> UnpackHDV(const HostDeviceVector<T>* _vec) const {
      return Span<T const> {_vec->ConstHostPointer(),
            static_cast<typename Span<T>::index_type>(_vec->Size())};
    }
    // Recursive sync host
    template <typename T>
    void SyncHost(const HostDeviceVector<T> *_vector) const {
      _vector->ConstHostPointer();
    }
    template <typename Head, typename... Rest>
    void SyncHost(const HostDeviceVector<Head> *_vector,
                  const HostDeviceVector<Rest> *... _vectors) const {
      _vector->ConstHostPointer();
      SyncHost(_vectors...);
    }
    // Recursive unpack for Shard.
    template <typename T>
    void UnpackShard(int device, const HostDeviceVector<T> *vector) const {
      vector->SetDevice(device);
    }
    template <typename Head, typename... Rest>
    void UnpackShard(int device,
                     const HostDeviceVector<Head> *_vector,
                     const HostDeviceVector<Rest> *... _vectors) const {
      _vector->SetDevice(device);
      UnpackShard(device, _vectors...);
    }

#if defined(__CUDACC__)
    template <typename std::enable_if<CompiledWithCuda>::type* = nullptr,
              typename... HDV>
    void LaunchCUDA(Functor _func, HDV*... _vectors) const {
      if (shard_) {
        UnpackShard(device_, _vectors...);
      }

      size_t range_size = *range_.end() - *range_.begin();

      // Extract index to deal with possible old OpenMP.
      // This deals with situation like multi-class setting where
      // granularity is used in data vector.
      size_t shard_size = range_size;
      Range shard_range {0, static_cast<Range::DifferenceType>(shard_size)};
      dh::safe_cuda(cudaSetDevice(device_));
      const int kGrids =
          static_cast<int>(DivRoundUp(*(range_.end()), kBlockThreads));
      if (kGrids == 0) {
        return;
      }
      detail::LaunchCUDAKernel<<<kGrids, kBlockThreads>>>(  // NOLINT
          _func, shard_range, UnpackHDVOnDevice(_vectors)...);
    }
#else
    /*! \brief Dummy funtion defined when compiling for CPU.  */
    template <typename std::enable_if<!CompiledWithCuda>::type* = nullptr,
              typename... HDV>
    void LaunchCUDA(Functor _func, HDV*... _vectors) const {
      LOG(FATAL) << "Not part of device code. WITH_CUDA: " << WITH_CUDA();
    }
#endif  // defined(__CUDACC__)

    template <typename... HDV>
    void LaunchCPU(Functor func, HDV*... vectors) const {
      omp_ulong end = static_cast<omp_ulong>(*(range_.end()));
      dmlc::OMPException omp_exc;
      SyncHost(vectors...);
#pragma omp parallel for schedule(static)
      for (omp_ulong idx = 0; idx < end; ++idx) {
        omp_exc.Run(func, idx, UnpackHDV(vectors)...);
      }
      omp_exc.Rethrow();
    }

   private:
    /*! \brief Callable object. */
    Functor func_;
    /*! \brief Range object specifying parallel threads index range. */
    Range range_;
    /*! \brief Whether sharding for vectors is required. */
    bool shard_;
    int device_;
  };

 public:
  /*!
   * \brief Initialize a Transform object.
   *
   * \tparam Functor  A callable object type.
   * \return A Evaluator having one method Eval.
   *
   * \param func    A callable object, accepting a size_t thread index,
   *                  followed by a set of Span classes.
   * \param range   Range object specifying parallel threads index range.
   * \param device  Specify GPU to use.
   * \param shard Whether Shard for HostDeviceVector is needed.
   */
  template <typename Functor>
  static Evaluator<Functor> Init(Functor func, Range const range,
                                 int device,
                                 bool const shard = true) {
    return Evaluator<Functor> {func, std::move(range), device, shard};
  }
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_TRANSFORM_H_
