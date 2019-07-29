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
  if (j_arr_obj.find("mask") != j_arr_obj.cend()) {
    LOG(FATAL) << "Meta info " << key << " should be dense, found validity mask";
  }

  float* p_data = reinterpret_cast<float*>(static_cast<size_t>(
      get<Integer const>(
          get<Array const>(
              j_arr_obj.at("data")).at(0))));

  auto version = get<Integer const>(j_arr_obj.at("version"));
  CHECK_EQ(version, 1) << "Only version 1 of __cuda_array_interface__ is being supported";

  auto typestr = get<String const>(j_arr_obj.at("typestr"));
  CHECK_EQ(typestr.size(),    3) << "`typestr` should be of format <endian><type><size>.";
  CHECK_NE(typestr.front(), '>') << "Big endian is not supported yet.";
  CHECK_EQ(typestr.at(2),   '4') << "Please convert the input into float32 first.";

  auto strides = get<Array const>(j_arr_obj.at("strides"));
  CHECK_EQ(get<Integer>(strides.at(0)), 4) << "Memory should be contigious.";

  auto j_shape = get<Array const>(j_arr_obj.at("shape"));
  CHECK_EQ(j_shape.size(), 1) << "Only 1 dimension column is valid.";
  auto length = get<Integer const>(j_shape.at(0));

  thrust::device_ptr<float> d_p_data {p_data};

  HostDeviceVector<float>* d_data;
  if (key == "root_index") {
    LOG(FATAL) << "root index for columnar data is not supported.";
  } else if (key == "label") {
    d_data = &labels_;
    CHECK_EQ(typestr.at(1),   'f') << "Label should be of floating point type.";
    CHECK_EQ(typestr.at(2),   '4') << "Please convert the input into float32 first.";
  } else if (key == "weight") {
    d_data = &weights_;
    CHECK_EQ(typestr.at(1),   'f') << "Weight should be of floating point type.";
    CHECK_EQ(typestr.at(2),   '4') << "Please convert the input into float32 first.";
  } else if (key == "base_margin") {
    d_data = &base_margin_;
    CHECK_EQ(typestr.at(1),   'f') << "Base Margin should be of floating point type.";
    CHECK_EQ(typestr.at(2),   '4') << "Please convert the input into float32 first.";
  } else if (key == "group") {
    CHECK_EQ(typestr.at(1),   'u') << "Group should be unsigned 32 bit integers.";
    CHECK_EQ(typestr.at(2),   '4') << "Please convert the Group into unsigned 32 bit integers first.";
    group_ptr_.resize(length + 1);
    group_ptr_[0] = 0;
    // Ranking is not performed on device.
    thrust::copy(d_p_data, d_p_data + length, group_ptr_.begin() + 1);
    for (size_t i = 1; i < group_ptr_.size(); ++i) {
      group_ptr_[i] = group_ptr_[i - 1] + group_ptr_[i];
    }
    return;
  } else {
    LOG(FATAL) << "Unknown metainfo: " << key;
  }
  thrust::copy(d_p_data, d_p_data + length, d_data->DeviceSpan(0).begin());
}
}  // namespace xgboost