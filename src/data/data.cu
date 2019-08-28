/*!
 * Copyright 2019 by XGBoost Contributors
 *
 * \file data.cu
 * \brief Handles setting metainfo from array interface.
 */
#include "xgboost/data.h"
#include "xgboost/logging.h"
#include "xgboost/json.h"
#include "columnar.h"
#include "../common/device_helpers.cuh"

namespace xgboost {

template <typename T>
void CopyInfoImpl(std::map<std::string, Json> const& column, HostDeviceVector<float>* out) {
  auto SetDeviceToPtr = [](void* ptr) {
    cudaPointerAttributes attr;
    dh::safe_cuda(cudaPointerGetAttributes(&attr, ptr));
    int32_t ptr_device = attr.device;
    dh::safe_cuda(cudaSetDevice(ptr_device));
    return ptr_device;
  };

  common::Span<T> s_data { ArrayInterfaceHandler::ExtractData<T>(column) };
  auto ptr_device = SetDeviceToPtr(s_data.data());
  thrust::device_ptr<T> p_src {s_data.data()};

  auto length = s_data.size();
  out->SetDevice(ptr_device);
  out->Resize(length);

  auto p_dst = thrust::device_pointer_cast(out->DevicePointer());
  thrust::copy(p_src, p_src + length, p_dst);
}

void CopyInfo(std::map<std::string, Json> const& column, HostDeviceVector<float>* out) {
  auto const& typestr = get<String const>(column.at("typestr"));
  if (typestr.at(1) == 'f' && typestr.at(2) == '4') {
    CopyInfoImpl<float>(column, out);
  } else if (typestr.at(1) == 'f' && typestr.at(2) == '8') {
    CopyInfoImpl<double>(column, out);
  } else if (typestr.at(1) == 'i' && typestr.at(2) == '1') {
    CopyInfoImpl<int8_t>(column, out);
  } else if (typestr.at(1) == 'i' && typestr.at(2) == '2') {
    CopyInfoImpl<int16_t>(column, out);
  } else if (typestr.at(1) == 'i' && typestr.at(2) == '4') {
    CopyInfoImpl<int32_t>(column, out);
  } else if (typestr.at(1) == 'i' && typestr.at(2) == '8') {
    CopyInfoImpl<int64_t>(column, out);
  } else if (typestr.at(1) == 'u' && typestr.at(2) == '1') {
    CopyInfoImpl<uint8_t>(column, out);
  } else if (typestr.at(1) == 'u' && typestr.at(2) == '2') {
    CopyInfoImpl<uint16_t>(column, out);
  } else if (typestr.at(1) == 'u' && typestr.at(2) == '4') {
    CopyInfoImpl<uint32_t>(column, out);
  } else if (typestr.at(1) == 'u' && typestr.at(2) == '8') {
    CopyInfoImpl<uint64_t>(column, out);
  } else {
    LOG(FATAL) << ColumnarErrors::UnknownTypeStr(typestr);
  }
}

void MetaInfo::SetInfo(const char * c_key, std::string const& interface_str) {
  Json j_arr = Json::Load({interface_str.c_str(), interface_str.size()});
  auto const& j_arr_obj = get<Object>(j_arr);
  std::string key {c_key};
  ArrayInterfaceHandler::Validate(j_arr_obj);
  if (j_arr_obj.find("mask") != j_arr_obj.cend()) {
    LOG(FATAL) << "Meta info " << key << " should be dense, found validity mask";
  }

  if (key == "root_index") {
    LOG(FATAL) << "root index for columnar data is not supported.";
  } else if (key == "label") {
    CopyInfo(j_arr_obj, &labels_);
  } else if (key == "weight") {
    CopyInfo(j_arr_obj, &weights_);
  } else if (key == "base_margin") {
    CopyInfo(j_arr_obj, &base_margin_);
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
}  // namespace xgboost
