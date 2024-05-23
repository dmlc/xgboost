/**
 * Copyright (c) 2023, XGBoost contributors
 */
#pragma once

#include <dmlc/registry.h>  // for Registry
#include <gtest/gtest.h>
#include <xgboost/objective.h>  // for ObjFunctionReg

#include <algorithm>  // for transform
#include <iterator>   // for back_insert_iterator, back_inserter
#include <string>     // for string
#include <vector>     // for vector

namespace xgboost {
inline auto MakeObjNamesForTest() {
  auto list = ::dmlc::Registry<::xgboost::ObjFunctionReg>::List();
  std::vector<std::string> names;
  std::transform(list.cbegin(), list.cend(), std::back_inserter(names),
                 [](auto const* entry) { return entry->name; });
  return names;
}

template <typename ParamType>
inline std::string ObjTestNameGenerator(const ::testing::TestParamInfo<ParamType>& info) {
  auto name = std::string{info.param};
  // Name must be a valid c++ symbol
  auto it = std::find(name.cbegin(), name.cend(), ':');
  if (it != name.cend()) {
    name[std::distance(name.cbegin(), it)] = '_';
  }
  return name;
};

/**
 * @brief Construct random label for testing.
 */
void MakeLabelForObjTest(std::shared_ptr<DMatrix> p_fmat, std::string const& obj);

std::shared_ptr<DMatrix> MakeFmatForObjTest(std::string const& obj, bst_idx_t n_samples,
                                            bst_feature_t n_features);
}  // namespace xgboost
