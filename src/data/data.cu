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

void CopyInfoImpl(std::map<std::string, Json> const& column, HostDeviceVector<float>* out) {
  auto SetDeviceToPtr = [](void* ptr) {
    cudaPointerAttributes attr;
    dh::safe_cuda(cudaPointerGetAttributes(&attr, ptr));
    int32_t ptr_device = attr.device;
    dh::safe_cuda(cudaSetDevice(ptr_device));
    return ptr_device;
  };
  ArrayInterface foreign_column(column);
  auto ptr_device = SetDeviceToPtr(foreign_column.data);

  out->SetDevice(ptr_device);
  out->Resize(foreign_column.num_rows);

  auto p_dst = thrust::device_pointer_cast(out->DevicePointer());

  dh::LaunchN(ptr_device, foreign_column.num_rows, [=] __device__(size_t idx) {
    p_dst[idx] = foreign_column.GetElement(idx);
  });
}

void MetaInfo::SetInfo(const char * c_key, std::string const& interface_str) {
  Json j_interface = Json::Load({interface_str.c_str(), interface_str.size()});
  auto const& j_arr = get<Array>(j_interface);
  CHECK_EQ(j_arr.size(), 1) << "MetaInfo: " << c_key << ". " << ArrayInterfaceErrors::Dimension(1);;
  auto const& j_arr_obj = get<Object const>(j_arr[0]);
  std::string key {c_key};
  ArrayInterfaceHandler::Validate(j_arr_obj);
  if (j_arr_obj.find("mask") != j_arr_obj.cend()) {
    LOG(FATAL) << "Meta info " << key << " should be dense, found validity mask";
  }
  auto const& typestr = get<String const>(j_arr_obj.at("typestr"));

  if (key == "label") {
    CopyInfoImpl(j_arr_obj, &labels_);
  } else if (key == "weight") {
    CopyInfoImpl(j_arr_obj, &weights_);
  } else if (key == "base_margin") {
    CopyInfoImpl(j_arr_obj, &base_margin_);
  } else if (key == "group") {
    // Ranking is not performed on device.
    auto s_data = ArrayInterfaceHandler::ExtractData<uint32_t>(j_arr_obj);
    thrust::device_ptr<uint32_t> p_src {s_data.data()};

    auto length = s_data.size();
    group_ptr_.resize(length + 1);
    group_ptr_[0] = 0;
    thrust::copy(p_src, p_src + length, group_ptr_.begin() + 1);
    std::partial_sum(group_ptr_.begin(), group_ptr_.end(), group_ptr_.begin());

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
