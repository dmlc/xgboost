/*!
 * Copyright 2019 by XGBoost Contributors
 *
 * \file data.cu
 */

#include "xgboost/data.h"
#include "xgboost/logging.h"
#include "xgboost/json.h"
#include "columnar.h"
#include "../common/device_helpers.cuh"

namespace xgboost {

void MetaInfo::SetInfo(const char * c_key, std::string const& interface_str) {
  Json j_arr = Json::Load({interface_str.c_str(), interface_str.size()});
  auto const& j_arr_obj = get<Object>(j_arr);
  std::string key {c_key};
  auto version = get<Integer const>(j_arr_obj.at("version"));
  CHECK_EQ(version, 1) << ColumnarErrors::Version();
  if (j_arr_obj.find("mask") != j_arr_obj.cend()) {
    LOG(FATAL) << "Meta info " << key << " should be dense, found validity mask";
  }

  auto typestr = get<String const>(j_arr_obj.at("typestr"));
  CHECK_EQ(typestr.size(),    3) << ColumnarErrors::TypestrFormat();
  CHECK_NE(typestr.front(), '>') << ColumnarErrors::BigEndian();

  auto j_shape = get<Array const>(j_arr_obj.at("shape"));
  CHECK_EQ(j_shape.size(), 1) << ColumnarErrors::Dimension(1);
  auto length = get<Integer const>(j_shape.at(0));
  CHECK_GT(length, 0) << "Label set cannot be empty.";

  if (j_arr_obj.find("strides") != j_arr_obj.cend()) {
    auto strides = get<Array const>(j_arr_obj.at("strides"));
    CHECK_EQ(get<Integer>(strides.at(0)), 4) << ColumnarErrors::Contigious();
  }

  float* p_data = GetPtrFromArrayData<float*>(j_arr_obj);

  cudaPointerAttributes attr;
  dh::safe_cuda(cudaPointerGetAttributes(&attr, p_data));
  int32_t ptr_device = attr.device;
  dh::safe_cuda(cudaSetDevice(ptr_device));

  thrust::device_ptr<float> p_src {p_data};

  HostDeviceVector<float>* dst;
  if (key == "root_index") {
    LOG(FATAL) << "root index for columnar data is not supported.";
  } else if (key == "label") {
    dst = &labels_;
    CHECK_EQ(typestr.at(1),   'f') << "Label"
                                   << ColumnarErrors::ofType("floating point");
    CHECK_EQ(typestr.at(2),   '4') << ColumnarErrors::toFloat();
  } else if (key == "weight") {
    dst = &weights_;
    CHECK_EQ(typestr.at(1),   'f') << "Weight"
                                   << ColumnarErrors::ofType("floating point");;
    CHECK_EQ(typestr.at(2),   '4') << ColumnarErrors::toFloat();
  } else if (key == "base_margin") {
    dst = &base_margin_;
    CHECK_EQ(typestr.at(1),   'f') << "Base Margin"
                                   << ColumnarErrors::ofType("floating point");
    CHECK_EQ(typestr.at(2),   '4') << ColumnarErrors::toFloat();
  } else if (key == "group") {
    CHECK_EQ(typestr.at(1),   'u') << "Group"
                                   << ColumnarErrors::ofType("unsigned 32 bit integers");
    CHECK_EQ(typestr.at(2),   '4') << ColumnarErrors::toUInt();
    group_ptr_.resize(length + 1);
    group_ptr_[0] = 0;
    // Ranking is not performed on device.
    thrust::copy(p_src, p_src + length, group_ptr_.begin() + 1);
    for (size_t i = 1; i < group_ptr_.size(); ++i) {
      group_ptr_[i] = group_ptr_[i - 1] + group_ptr_[i];
    }
    return;
  } else {
    LOG(FATAL) << "Unknown metainfo: " << key;
  }
  dst->Reshard(ptr_device);
  dst->Resize(length);
  auto p_dst = thrust::device_pointer_cast(dst->DevicePointer(0));
  thrust::copy(p_src, p_src + length, p_dst);
}
}  // namespace xgboost
