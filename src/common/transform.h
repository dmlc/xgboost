/*!
 * Copyright 2018 XGBoost contributors
 */
#ifndef XGBOOST_COMMON_TRANSFORM_H_
#define XGBOOST_COMMON_TRANSFORM_H_

#include <dmlc/omp.h>
#include <vector>
#include <type_traits>

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
#if defined (__CUDACC__)

template <typename T>
__device__ Range SegGridStrideRange(T _end_it, int64_t _segment) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  idx *= _segment;
  if (idx >= *_end_it) {
    return Range {0, 0};
  }
  return Range{idx, *_end_it, static_cast<int64_t>(gridDim.x * blockDim.x)};
}

template <typename T>
__device__ Span<T> Unpack(Span<T> _span, size_t _idx, size_t _size) {
  // dealing with empty weight.
  return _span.size() == 0 ? _span : _span.subspan(_idx, _size);
}


template <typename Functor, typename... T>
__global__ void LaunchRangeCUDAKernel(Functor _func, Range _range,
                                      Span<T> ... _spans) {
  for (auto i : SegGridStrideRange(_range.end(), _range.GetStep())) {
    _func(Unpack(_spans, i, _range.GetStep())...);
  }
}

template <typename Functor, typename U, typename... T>
__global__ void LaunchRangeCUDAKernel(Functor _func, Range _range,
                                      U* _flag, Span<T> ... _spans) {
  for (auto i : SegGridStrideRange(_range.end(), _range.GetStep())) {
    _func(_flag, Unpack(_spans, i, _range.GetStep())...);
  }
}
#endif

}  // namespace detail


/*
 * \brief Transform HostDeviceVectors with segmented data.
 *
 * Note for users:
 *   To handle empty weight, zero sized vector will be passed to every
 *   instance of functor as it's.
 *
 * Two form of SegTransform, one with flags, which represents a HostDeviceVector
 * having one element for each device. Another form is without flags.
 * Code duplication is a drawback.  If we want to dispatch at runtime, user must
 * define a `flag` parameter in input functor even if it is not used.
 */
struct SegTransform {
  /*
   * \brief Segmented transform.
   *
   * \tparam Functor Callable object type, accepting spans expanded from
   *                   varadic HostDeviceVector.
   * \tparam T       Numeric type of Data.
   *
   * \param func     Instance of Functor.
   * \param range    A Range object specifying begin, end and step (segment size).
   * \param devices  GPUSet for this function.
   * \param vectors  Varadic HostDeviceVector<T>, expanded into a set of
   *                   Span<T> as input for func.
   */
  template <typename Functor, typename... T>
  SegTransform(Functor _func, Range _range, GPUSet _devices,
               HostDeviceVector<T>*... _vectors) {
    CHECK((*(_range.end()) - *(_range.begin()) + 1) % _range.GetStep() == 0) <<
        "Segment length must divides total size.";

    Reshard(_devices, _vectors...);

    bool on_device = _devices != GPUSet::Empty();
    if (on_device) {
#if defined(__CUDACC__)
      LaunchRangeCUDA(_func, _range, _devices, _vectors...);
#else
      LOG(FATAL) << "Not part of device code.";
#endif
    } else {
      LaunchRangeCPU(_func, _range, UnpackHDV(_vectors)...);
    }
  }

  /*
   * \brief Segmented transform.
   *
   * \tparam Functor Callable object type, accepting spans expanded from
   *                   varadic HostDeviceVector.
   * \tparam T       Numeric type of Data.
   *
   * \param func     Instance of Functor.
   * \param range    A Range object specifying begin, end and step (segment size).
   * \param flags    A pointer to HostDeviceVector, with one element for each
   *                   device.  Passed to func as pointer to element.
   * \param devices  GPUSet for this function.
   * \param vectors  Varadic HostDeviceVector<T>, expanded into a set of
   *                   Span<T> as input for func.
   */
  template <typename Functor, typename U, typename... T>
  SegTransform(Functor _func, Range _range, HostDeviceVector<U>* _flags,
               GPUSet _devices, HostDeviceVector<T>*... _vectors) {
    CHECK((*(_range.end()) - *(_range.begin()) + 1) % _range.GetStep() == 0) <<
        "Segment length must divides total size.";

    Reshard(_devices, _vectors...);

    bool on_device = _devices != GPUSet::Empty();
    if (on_device) {
#if defined(__CUDACC__)
      LaunchRangeCUDA(_func, _range, _flags, _devices, _vectors...);
#else
      LOG(FATAL) << "Not part of device code.";
#endif
    } else {
      LaunchRangeCPU(_func, _range, _flags->HostPointer(), UnpackHDV(_vectors)...);
    }
  }

 private:
  template <typename... T>
  void Reshard(GPUSet _devices, HostDeviceVector<T>*... _vectors) {
    std::vector<HDVAny> vectors {_vectors...};
#pragma omp parallel for schedule(static, 1) if (_devices.Size() > 1)
    for (omp_ulong i = 0; i < vectors.size(); ++i) {  // NOLINT
      switch (vectors[i].GetType()) {
        case HDVAny::Type::kBstFloatType:
          vectors[i].GetFloat()->Reshard(_devices);
          break;
        case HDVAny::Type::kGradientPairType:
          vectors[i].GetGradientPair()->Reshard(_devices);
          break;
        case HDVAny::Type::kUIntType:
          vectors[i].GetUnsignedInt()->Reshard(_devices);
        default:
          LOG(FATAL) << "Unknown HostDeviceVector type.";
      }
    }
  }

  template <typename T>
  Span<T> UnpackHDV(HostDeviceVector<T>* _vec, int _device) {
    return _vec->DeviceSpan(_device);
  }

  template <typename T>
  Span<T> UnpackHDV(HostDeviceVector<T>* _vec) {
    return Span<T> {_vec->HostPointer(),
          static_cast<typename Span<T>::index_type>(_vec->Size())};
  }
  template <typename T>
  Span<T> UnpackSubspan(Span<T> _span, int64_t _offset, int64_t _step) {
    // dealing with empty weight.
    if (_span.size() == 0) {
      return _span;
    }
    return _span.subspan(_offset, _step);
  }

#if defined(__CUDACC__)

  template <typename Functor, typename... T>
  void LaunchRangeCUDA(Functor _func, Range _range, GPUSet _devices,
                       HostDeviceVector<T>*... _vectors) {
#pragma omp parallel for schedule(static, 1) if (_devices.Size() > 1)
    for (omp_ulong i = 0; i < _devices.Size(); ++i) {
      int d = GPUSet::GetDeviceIdx(_devices[i]);
      dh::safe_cuda(cudaSetDevice(d));
      const int GRID_SIZE =
          static_cast<int>(dh::DivRoundUp(*(_range.end()),
                                          _range.GetStep() * kBlockThreads));

      detail::LaunchRangeCUDAKernel<<<GRID_SIZE, kBlockThreads>>>(
          _func, _range, UnpackHDV(_vectors, d)...);
      dh::safe_cuda(cudaGetLastError());
      dh::safe_cuda(cudaDeviceSynchronize());
    }
  }

  template <typename Functor, typename U, typename... T>
  void LaunchRangeCUDA(Functor _func, Range _range, HostDeviceVector<U>* _flags,
                       GPUSet _devices, HostDeviceVector<T>*... _vectors) {
#pragma omp parallel for schedule(static, 1) if (_devices.Size() > 1)
    for (omp_ulong i = 0; i < _devices.Size(); ++i) {
      int d = GPUSet::GetDeviceIdx(_devices[i]);
      dh::safe_cuda(cudaSetDevice(d));
      const int GRID_SIZE =
          static_cast<int>(dh::DivRoundUp(*(_range.end()),
                                          _range.GetStep() * kBlockThreads));

      detail::LaunchRangeCUDAKernel<<<GRID_SIZE, kBlockThreads>>>(
          _func, _range, _flags->DevicePointer(d), UnpackHDV(_vectors, d)...);
      dh::safe_cuda(cudaGetLastError());
      dh::safe_cuda(cudaDeviceSynchronize());
    }
  }

#endif

  template <typename Functor, typename... T>
  void LaunchRangeCPU(Functor _func, common::Range _range, Span<T> ... _spans) {
    auto end = *(_range.end());
    auto step = _range.GetStep();
#pragma omp parallel for schedule(static)
    for (omp_ulong idx = 0; idx < end; idx += step) {
      _func(UnpackSubspan(_spans, idx, step)...);
    }
  }

  template <typename Functor, typename U, typename... T>
  void LaunchRangeCPU(Functor _func, common::Range _range, U* _flag,
                      Span<T> ... _spans) {
    auto end = *(_range.end());
    auto step = _range.GetStep();
#pragma omp parallel for schedule(static)
    for (omp_ulong idx = 0; idx < end; idx += step) {
      _func(_flag, UnpackSubspan(_spans, idx, step)...);
    }
  }
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_TRANSFORM_H_
