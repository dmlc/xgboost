/*!
 * Copyright 2019-2021 by XGBoost Contributors
 *
 * \file data.cu
 * \brief Handles setting metainfo from array interface.
 */
#include "xgboost/data.h"
#include "xgboost/logging.h"
#include "xgboost/json.h"
#include "array_interface.h"
#include "../common/device_helpers.cuh"
#include "device_adapter.cuh"
#include "simple_dmatrix.h"

namespace xgboost {

void CopyInfoImpl(ArrayInterface column, HostDeviceVector<float>* out) {
  auto SetDeviceToPtr = [](void* ptr) {
    cudaPointerAttributes attr;
    dh::safe_cuda(cudaPointerGetAttributes(&attr, ptr));
    int32_t ptr_device = attr.device;
    dh::safe_cuda(cudaSetDevice(ptr_device));
    return ptr_device;
  };
  auto ptr_device = SetDeviceToPtr(column.data);

  out->SetDevice(ptr_device);
  out->Resize(column.num_rows);

  auto p_dst = thrust::device_pointer_cast(out->DevicePointer());

  dh::LaunchN(ptr_device, column.num_rows, [=] __device__(size_t idx) {
    p_dst[idx] = column.GetElement(idx, 0);
  });
}

namespace {
auto SetDeviceToPtr(void *ptr) {
  cudaPointerAttributes attr;
  dh::safe_cuda(cudaPointerGetAttributes(&attr, ptr));
  int32_t ptr_device = attr.device;
  dh::safe_cuda(cudaSetDevice(ptr_device));
  return ptr_device;
}
}  // anonymous namespace

void CopyGroupInfoImpl(ArrayInterface column, std::vector<bst_group_t>* out) {
  CHECK(column.type != ArrayInterface::kF4 && column.type != ArrayInterface::kF8)
      << "Expected integer for group info.";

  auto ptr_device = SetDeviceToPtr(column.data);
  dh::TemporaryArray<bst_group_t> temp(column.num_rows);
  auto d_tmp = temp.data();

  dh::LaunchN(ptr_device, column.num_rows, [=] __device__(size_t idx) {
    d_tmp[idx] = column.GetElement<size_t>(idx, 0);
  });
  auto length = column.num_rows;
  out->resize(length + 1);
  out->at(0) = 0;
  thrust::copy(temp.data(), temp.data() + length, out->begin() + 1);
  std::partial_sum(out->begin(), out->end(), out->begin());
}

void CopyQidImpl(ArrayInterface array_interface,
                 std::vector<bst_group_t> *p_group_ptr) {
  auto &group_ptr_ = *p_group_ptr;
  auto it = dh::MakeTransformIterator<uint32_t>(
      thrust::make_counting_iterator(0ul),
      [array_interface] __device__(size_t i) {
        return array_interface.GetElement<uint32_t>(i, 0);
      });
  dh::caching_device_vector<bool> flag(1);
  auto d_flag = dh::ToSpan(flag);
  auto d = SetDeviceToPtr(array_interface.data);
  dh::LaunchN(d, 1, [=] __device__(size_t) { d_flag[0] = true; });
  dh::LaunchN(d, array_interface.num_rows - 1, [=] __device__(size_t i) {
    if (array_interface.GetElement<uint32_t>(i, 0) >
        array_interface.GetElement<uint32_t>(i + 1, 0)) {
      d_flag[0] = false;
    }
  });
  bool non_dec = true;
  dh::safe_cuda(cudaMemcpy(&non_dec, flag.data().get(), sizeof(bool),
                           cudaMemcpyDeviceToHost));
  CHECK(non_dec) << "`qid` must be sorted in increasing order along with data.";
  size_t bytes = 0;
  dh::caching_device_vector<uint32_t> out(array_interface.num_rows);
  dh::caching_device_vector<uint32_t> cnt(array_interface.num_rows);
  HostDeviceVector<int> d_num_runs_out(1, 0, d);
  cub::DeviceRunLengthEncode::Encode(
      nullptr, bytes, it, out.begin(), cnt.begin(),
      d_num_runs_out.DevicePointer(), array_interface.num_rows);
  dh::caching_device_vector<char> tmp(bytes);
  cub::DeviceRunLengthEncode::Encode(
      tmp.data().get(), bytes, it, out.begin(), cnt.begin(),
      d_num_runs_out.DevicePointer(), array_interface.num_rows);

  auto h_num_runs_out = d_num_runs_out.HostSpan()[0];
  group_ptr_.clear();
  group_ptr_.resize(h_num_runs_out + 1, 0);
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::inclusive_scan(thrust::cuda::par(alloc), cnt.begin(),
                         cnt.begin() + h_num_runs_out, cnt.begin());
  thrust::copy(cnt.begin(), cnt.begin() + h_num_runs_out,
               group_ptr_.begin() + 1);
}

namespace {
// thrust::all_of tries to copy lambda function.
struct AllOfOp {
  __device__ bool operator()(float w) {
    return w >= 0;
  }
};
}  // anonymous namespace

void MetaInfo::SetInfo(const char * c_key, std::string const& interface_str) {
  Json j_interface = Json::Load({interface_str.c_str(), interface_str.size()});
  auto const& j_arr = get<Array>(j_interface);
  CHECK_EQ(j_arr.size(), 1)
      << "MetaInfo: " << c_key << ". " << ArrayInterfaceErrors::Dimension(1);
  ArrayInterface array_interface(interface_str);
  std::string key{c_key};
  array_interface.AsColumnVector();
  CHECK(!array_interface.valid.Data())
      << "Meta info " << key << " should be dense, found validity mask";
  if (array_interface.num_rows == 0) {
    return;
  }

  if (key == "label") {
    CopyInfoImpl(array_interface, &labels_);
  } else if (key == "weight") {
    CopyInfoImpl(array_interface, &weights_);
    auto ptr = weights_.ConstDevicePointer();
    auto valid =
        thrust::all_of(thrust::device, ptr, ptr + weights_.Size(), AllOfOp{});
    CHECK(valid) << "Weights must be positive values.";
  } else if (key == "base_margin") {
    CopyInfoImpl(array_interface, &base_margin_);
  } else if (key == "group") {
    CopyGroupInfoImpl(array_interface, &group_ptr_);
    return;
  } else if (key == "qid") {
    CopyQidImpl(array_interface, &group_ptr_);
    return;
  } else if (key == "label_lower_bound") {
    CopyInfoImpl(array_interface, &labels_lower_bound_);
    return;
  } else if (key == "label_upper_bound") {
    CopyInfoImpl(array_interface, &labels_upper_bound_);
    return;
  } else if (key == "feature_weights") {
    CopyInfoImpl(array_interface, &feature_weigths);
    auto d_feature_weights = feature_weigths.ConstDeviceSpan();
    auto valid = thrust::all_of(
        thrust::device, d_feature_weights.data(),
        d_feature_weights.data() + d_feature_weights.size(), AllOfOp{});
    CHECK(valid) << "Feature weight must be greater than 0.";
    return;
  } else {
    LOG(FATAL) << "Unknown metainfo: " << key;
  }
}

template <typename AdapterT>
DMatrix* DMatrix::Create(AdapterT* adapter, float missing, int nthread,
                         const std::string& cache_prefix, size_t page_size) {
  CHECK_EQ(cache_prefix.size(), 0)
      << "Device memory construction is not currently supported with external "
         "memory.";
  return new data::SimpleDMatrix(adapter, missing, nthread);
}

template DMatrix* DMatrix::Create<data::CudfAdapter>(
    data::CudfAdapter* adapter, float missing, int nthread,
    const std::string& cache_prefix, size_t page_size);
template DMatrix* DMatrix::Create<data::CupyAdapter>(
    data::CupyAdapter* adapter, float missing, int nthread,
    const std::string& cache_prefix, size_t page_size);
}  // namespace xgboost
