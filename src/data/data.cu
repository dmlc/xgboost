/*!
 * Copyright 2019 by XGBoost Contributors
 *
 * \file data.cu
 * \brief Handles setting metainfo from array interface.
 */
#include "xgboost/data.h"
#include "xgboost/logging.h"

#include "columnar.h"
#include "../common/device_helpers.cuh"
#include "../common/json_experimental.h"
#include "../common/json_reader_experimental.h"

namespace xgboost {

template <typename T>
void CopyInfoImpl(experimental::Json const& column, HostDeviceVector<float>* out) {
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

void MetaInfo::SetInfo(const char * c_key, std::string const& interface_str) {
  std::string copied = interface_str;
  experimental::Document j_interface =
      experimental::Document::Load<experimental::JsonRecursiveReader>(
          experimental::StringRef{copied});
  auto const& j_arr = j_interface.GetValue();
  CHECK_EQ(j_arr.Length(), 1) << "MetaInfo: " << c_key << ". " << ColumnarErrors::Dimension(1);;
  auto const& j_arr_obj = j_arr.GetArrayElem(0);
  std::string key {c_key};
  ArrayInterfaceHandler::Validate(j_arr_obj);
  if (j_arr_obj.FindMemberByKey("mask") != j_arr_obj.cend()) {
    LOG(FATAL) << "Meta info " << key << " should be dense, found validity mask";
  }
  auto const& typestr = (*j_arr_obj.FindMemberByKey("typestr")).GetString();

  if (key == "root_index") {
    LOG(FATAL) << "root index for columnar data is not supported.";
  } else if (key == "label") {
    DISPATCH_TYPE(CopyInfoImpl, typestr, j_arr_obj, &labels_);
  } else if (key == "weight") {
    DISPATCH_TYPE(CopyInfoImpl, typestr, j_arr_obj, &weights_);
  } else if (key == "base_margin") {
    DISPATCH_TYPE(CopyInfoImpl, typestr, j_arr_obj, &base_margin_);
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
