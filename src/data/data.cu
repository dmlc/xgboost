/**
 * Copyright 2019-2022 by XGBoost Contributors
 *
 * \file data.cu
 * \brief Handles setting metainfo from array interface.
 */
#include "../common/cuda_context.cuh"
#include "../common/device_helpers.cuh"
#include "../common/linalg_op.cuh"
#include "array_interface.h"
#include "device_adapter.cuh"
#include "simple_dmatrix.h"
#include "validation.h"
#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/logging.h"

namespace xgboost {
namespace {
auto SetDeviceToPtr(void const* ptr) {
  cudaPointerAttributes attr;
  dh::safe_cuda(cudaPointerGetAttributes(&attr, ptr));
  int32_t ptr_device = attr.device;
  dh::safe_cuda(cudaSetDevice(ptr_device));
  return ptr_device;
}

template <typename T, int32_t D>
void CopyTensorInfoImpl(CUDAContext const* ctx, Json arr_interface, linalg::Tensor<T, D>* p_out) {
  ArrayInterface<D> array(arr_interface);
  if (array.n == 0) {
    p_out->SetDevice(0);
    p_out->Reshape(array.shape);
    return;
  }
  CHECK(array.valid.Size() == 0) << "Meta info like label or weight can not have missing value.";
  auto ptr_device = SetDeviceToPtr(array.data);
  p_out->SetDevice(ptr_device);

  if (array.is_contiguous && array.type == ToDType<T>::kType) {
    p_out->ModifyInplace([&](HostDeviceVector<T>* data, common::Span<size_t, D> shape) {
      // set shape
      std::copy(array.shape, array.shape + D, shape.data());
      // set data
      data->Resize(array.n);
      dh::safe_cuda(cudaMemcpyAsync(data->DevicePointer(), array.data, array.n * sizeof(T),
                                    cudaMemcpyDefault, ctx->Stream()));
    });
    return;
  }
  p_out->Reshape(array.shape);
  auto t = p_out->View(ptr_device);
  linalg::ElementWiseTransformDevice(
      t,
      [=] __device__(size_t i, T) {
        return linalg::detail::Apply(TypedIndex<T, D>{array},
                                     linalg::UnravelIndex<D>(i, array.shape));
      },
      ctx->Stream());
}

void CopyGroupInfoImpl(ArrayInterface<1> column, std::vector<bst_group_t>* out) {
  CHECK(column.type != ArrayInterfaceHandler::kF4 && column.type != ArrayInterfaceHandler::kF8)
      << "Expected integer for group info.";

  auto ptr_device = SetDeviceToPtr(column.data);
  CHECK_EQ(ptr_device, dh::CurrentDevice());
  dh::TemporaryArray<bst_group_t> temp(column.Shape(0));
  auto d_tmp = temp.data().get();

  dh::LaunchN(column.Shape(0),
              [=] __device__(size_t idx) { d_tmp[idx] = TypedIndex<size_t, 1>{column}(idx); });
  auto length = column.Shape(0);
  out->resize(length + 1);
  out->at(0) = 0;
  thrust::copy(temp.data(), temp.data() + length, out->begin() + 1);
  std::partial_sum(out->begin(), out->end(), out->begin());
}

void CopyQidImpl(ArrayInterface<1> array_interface, std::vector<bst_group_t>* p_group_ptr) {
  auto &group_ptr_ = *p_group_ptr;
  auto it = dh::MakeTransformIterator<uint32_t>(
      thrust::make_counting_iterator(0ul), [array_interface] __device__(size_t i) {
        return TypedIndex<uint32_t, 1>{array_interface}(i);
      });
  dh::caching_device_vector<bool> flag(1);
  auto d_flag = dh::ToSpan(flag);
  auto d = SetDeviceToPtr(array_interface.data);
  dh::LaunchN(1, [=] __device__(size_t) { d_flag[0] = true; });
  dh::LaunchN(array_interface.Shape(0) - 1, [=] __device__(size_t i) {
    auto typed = TypedIndex<uint32_t, 1>{array_interface};
    if (typed(i) > typed(i + 1)) {
      d_flag[0] = false;
    }
  });
  bool non_dec = true;
  dh::safe_cuda(cudaMemcpy(&non_dec, flag.data().get(), sizeof(bool),
                           cudaMemcpyDeviceToHost));
  CHECK(non_dec) << "`qid` must be sorted in increasing order along with data.";
  size_t bytes = 0;
  dh::caching_device_vector<uint32_t> out(array_interface.Shape(0));
  dh::caching_device_vector<uint32_t> cnt(array_interface.Shape(0));
  HostDeviceVector<int> d_num_runs_out(1, 0, d);
  cub::DeviceRunLengthEncode::Encode(
      nullptr, bytes, it, out.begin(), cnt.begin(),
      d_num_runs_out.DevicePointer(), array_interface.Shape(0));
  dh::caching_device_vector<char> tmp(bytes);
  cub::DeviceRunLengthEncode::Encode(
      tmp.data().get(), bytes, it, out.begin(), cnt.begin(),
      d_num_runs_out.DevicePointer(), array_interface.Shape(0));

  auto h_num_runs_out = d_num_runs_out.HostSpan()[0];
  group_ptr_.clear();
  group_ptr_.resize(h_num_runs_out + 1, 0);
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::inclusive_scan(thrust::cuda::par(alloc), cnt.begin(),
                         cnt.begin() + h_num_runs_out, cnt.begin());
  thrust::copy(cnt.begin(), cnt.begin() + h_num_runs_out,
               group_ptr_.begin() + 1);
}
}  // namespace

void MetaInfo::SetInfoFromCUDA(Context const& ctx, StringView key, Json array) {
  // multi-dim float info
  if (key == "base_margin") {
    CopyTensorInfoImpl(ctx.CUDACtx(), array, &base_margin_);
    return;
  } else if (key == "label") {
    CopyTensorInfoImpl(ctx.CUDACtx(), array, &labels);
    auto ptr = labels.Data()->ConstDevicePointer();
    auto valid = thrust::none_of(thrust::device, ptr, ptr + labels.Size(), data::LabelsCheck{});
    CHECK(valid) << "Label contains NaN, infinity or a value too large.";
    return;
  }
  // uint info
  if (key == "group") {
    ArrayInterface<1> array_interface{array};
    CopyGroupInfoImpl(array_interface, &group_ptr_);
    data::ValidateQueryGroup(group_ptr_);
    return;
  } else if (key == "qid") {
    ArrayInterface<1> array_interface{array};
    CopyQidImpl(array_interface, &group_ptr_);
    data::ValidateQueryGroup(group_ptr_);
    return;
  }
  // float info
  linalg::Tensor<float, 1> t;
  CopyTensorInfoImpl(ctx.CUDACtx(), array, &t);
  if (key == "weight") {
    this->weights_ = std::move(*t.Data());
    auto ptr = weights_.ConstDevicePointer();
    auto valid = thrust::none_of(thrust::device, ptr, ptr + weights_.Size(), data::WeightsCheck{});
    CHECK(valid) << "Weights must be positive values.";
  } else if (key == "label_lower_bound") {
    this->labels_lower_bound_ = std::move(*t.Data());
  } else if (key == "label_upper_bound") {
    this->labels_upper_bound_ = std::move(*t.Data());
  } else if (key == "feature_weights") {
    this->feature_weights = std::move(*t.Data());
    auto d_feature_weights = feature_weights.ConstDeviceSpan();
    auto valid =
        thrust::none_of(ctx.CUDACtx()->CTP(), d_feature_weights.data(),
                        d_feature_weights.data() + d_feature_weights.size(), data::WeightsCheck{});
    CHECK(valid) << "Feature weight must be greater than 0.";
  } else {
    LOG(FATAL) << "Unknown key for MetaInfo: " << key;
  }
}

template <typename AdapterT>
DMatrix* DMatrix::Create(AdapterT* adapter, float missing, int nthread,
                         const std::string& cache_prefix, DataSplitMode data_split_mode) {
  CHECK_EQ(cache_prefix.size(), 0)
      << "Device memory construction is not currently supported with external "
         "memory.";
  return new data::SimpleDMatrix(adapter, missing, nthread, data_split_mode);
}

template DMatrix* DMatrix::Create<data::CudfAdapter>(
    data::CudfAdapter* adapter, float missing, int nthread,
    const std::string& cache_prefix, DataSplitMode data_split_mode);
template DMatrix* DMatrix::Create<data::CupyAdapter>(
    data::CupyAdapter* adapter, float missing, int nthread,
    const std::string& cache_prefix, DataSplitMode data_split_mode);
}  // namespace xgboost
