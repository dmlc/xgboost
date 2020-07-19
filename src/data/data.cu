/*!
 * Copyright 2019 by XGBoost Contributors
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
    p_dst[idx] = column.GetElement(idx);
  });
}

void CopyGroupInfoImpl(ArrayInterface column, std::vector<bst_group_t>* out) {
  CHECK(column.type[1] == 'i' || column.type[1] == 'u')
      << "Expected integer metainfo";
  auto SetDeviceToPtr = [](void* ptr) {
    cudaPointerAttributes attr;
    dh::safe_cuda(cudaPointerGetAttributes(&attr, ptr));
    int32_t ptr_device = attr.device;
    dh::safe_cuda(cudaSetDevice(ptr_device));
    return ptr_device;
  };
  auto ptr_device = SetDeviceToPtr(column.data);
  dh::TemporaryArray<bst_group_t> temp(column.num_rows);
  auto d_tmp = temp.data();

  dh::LaunchN(ptr_device, column.num_rows, [=] __device__(size_t idx) {
    d_tmp[idx] = column.GetElement(idx);
  });
  auto length = column.num_rows;
  out->resize(length + 1);
  out->at(0) = 0;
  thrust::copy(temp.data(), temp.data() + length, out->begin() + 1);
  std::partial_sum(out->begin(), out->end(), out->begin());
}

void MetaInfo::SetInfo(const char * c_key, std::string const& interface_str) {
  Json j_interface = Json::Load({interface_str.c_str(), interface_str.size()});
  auto const& j_arr = get<Array>(j_interface);
  CHECK_EQ(j_arr.size(), 1)
      << "MetaInfo: " << c_key << ". " << ArrayInterfaceErrors::Dimension(1);
  ArrayInterface array_interface(interface_str);
  std::string key{c_key};
  CHECK(!array_interface.valid.Data())
      << "Meta info " << key << " should be dense, found validity mask";
  CHECK_EQ(array_interface.num_cols, 1)
      << "Meta info should be a single column.";
  if (array_interface.num_rows == 0) {
    return;
  }

  if (key == "label") {
    CopyInfoImpl(array_interface, &labels_);
  } else if (key == "weight") {
    CopyInfoImpl(array_interface, &weights_);
  } else if (key == "base_margin") {
    CopyInfoImpl(array_interface, &base_margin_);
  } else if (key == "group") {
    CopyGroupInfoImpl(array_interface, &group_ptr_);
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
