/*!
 * Copyright 2018 XGBoost contributors
 */
#ifndef XGBOOST_COMMON_TRANSFORM_H_
#define XGBOOST_COMMON_TRANSFORM_H_

#include <dmlc/omp.h>
#include <xgboost/data.h>
#include <vector>
#include <type_traits>  // enable_if

#include "host_device_vector.h"
#include "common.h"
#include "span.h"

#if defined (__CUDACC__)
#include "device_helpers.cuh"
#endif

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
#endif

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
    Evaluator(Functor func, Range range, GPUSet devices, bool reshard) :
        func_(func), range_{std::move(range)},
        distribution_{std::move(GPUDistribution::Block(devices))},
        reshard_{reshard} {}
    Evaluator(Functor func, Range range, GPUDistribution dist,
              bool reshard) :
        func_(func), range_{std::move(range)}, distribution_{std::move(dist)},
        reshard_{reshard} {}

    /*!
     * \brief Evaluate the functor with input pointers to HostDeviceVector.
     *
     * \tparam HDV...  HostDeviceVectors type.
     * \param  vectors Pointers to HostDeviceVector.
     */
    template <typename... HDV>
    void Eval(HDV... vectors) const {
      bool on_device = !distribution_.IsEmpty();

      if (on_device) {
        LaunchCUDA(func_, vectors...);
      } else {
        LaunchCPU(func_, vectors...);
      }
    }

   private:
    // CUDA UnpackHDV
    template <typename T>
    Span<T> UnpackHDV(HostDeviceVector<T>* _vec, int _device) const {
      return _vec->DeviceSpan(_device);
    }
    template <typename T>
    Span<T const> UnpackHDV(const HostDeviceVector<T>* _vec, int _device) const {
      return _vec->ConstDeviceSpan(_device);
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
    // Recursive unpack for Reshard.
    template <typename T>
    void UnpackReshard(GPUDistribution dist, const HostDeviceVector<T>* vector) const {
      vector->Reshard(dist);
    }
    template <typename Head, typename... Rest>
    void UnpackReshard(GPUDistribution dist,
                       const HostDeviceVector<Head>* _vector,
                       const HostDeviceVector<Rest>*... _vectors) const {
      _vector->Reshard(dist);
      UnpackReshard(dist, _vectors...);
    }

#if defined(__CUDACC__)
    template <typename std::enable_if<CompiledWithCuda>::type* = nullptr,
              typename... HDV>
    void LaunchCUDA(Functor _func, HDV*... _vectors) const {
      if (reshard_)
        UnpackReshard(distribution_, _vectors...);

      GPUSet devices = distribution_.Devices();
      size_t range_size = *range_.end() - *range_.begin();
#pragma omp parallel for schedule(static, 1) if (devices.Size() > 1)
      for (omp_ulong i = 0; i < devices.Size(); ++i) {
        int d = devices.Index(i);
        // Ignore other attributes of GPUDistribution for spliting index.
        size_t shard_size =
            GPUDistribution::Block(devices).ShardSize(range_size, d);
        Range shard_range {0, static_cast<Range::DifferenceType>(shard_size)};
        dh::safe_cuda(cudaSetDevice(d));
        const int GRID_SIZE =
            static_cast<int>(dh::DivRoundUp(*(range_.end()), kBlockThreads));

        detail::LaunchCUDAKernel<<<GRID_SIZE, kBlockThreads>>>(
            _func, shard_range, UnpackHDV(_vectors, d)...);
        dh::safe_cuda(cudaGetLastError());
        dh::safe_cuda(cudaDeviceSynchronize());
      }
    }
#else
    /*! \brief Dummy funtion defined when compiling for CPU.  */
    template <typename std::enable_if<!CompiledWithCuda>::type* = nullptr,
              typename... HDV>
    void LaunchCUDA(Functor _func, HDV*... _vectors) const {
      LOG(FATAL) << "Not part of device code. WITH_CUDA: " << WITH_CUDA();
    }
#endif

    template <typename... HDV>
    void LaunchCPU(Functor func, HDV*... vectors) const {
      auto end = *(range_.end());
#pragma omp parallel for schedule(static)
      for (omp_ulong idx = 0; idx < end; ++idx) {
        func(idx, UnpackHDV(vectors)...);
      }
    }

   private:
    /*! \brief Callable object. */
    Functor func_;
    /*! \brief Range object specifying parallel threads index range. */
    Range range_;
    /*! \brief Whether resharding for vectors is required. */
    bool reshard_;
    GPUDistribution distribution_;
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
   * \param devices GPUSet specifying GPUs to use, when compiling for CPU,
   *                  this should be GPUSet::Empty().
   * \param reshard Whether Reshard for HostDeviceVector is needed.
   */
  template <typename Functor>
  static Evaluator<Functor> Init(Functor func, Range const range,
                                 GPUSet const devices,
                                 bool const reshard = true) {
    return Evaluator<Functor> {func, std::move(range), std::move(devices), reshard};
  }
  template <typename Functor>
  static Evaluator<Functor> Init(Functor func, Range const range,
                                 GPUDistribution const dist,
                                 bool const reshard = true) {
    return Evaluator<Functor> {func, std::move(range), std::move(dist), reshard};
  }
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_TRANSFORM_H_
