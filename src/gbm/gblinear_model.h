/*!
 * Copyright by Contributors 2018
 */
#pragma once
#include <dmlc/io.h>
#include <dmlc/parameter.h>
#include <vector>
#include <cstring>

namespace xgboost {
namespace gbm {
// model parameter
struct GBLinearModelParam : public dmlc::Parameter<GBLinearModelParam> {
  // number of feature dimension
  unsigned num_feature;
  // number of output group
  int num_output_group;
  // reserved field
  int reserved[32];
  // constructor
  GBLinearModelParam() { std::memset(this, 0, sizeof(GBLinearModelParam)); }
  DMLC_DECLARE_PARAMETER(GBLinearModelParam) {
    DMLC_DECLARE_FIELD(num_feature)
        .set_lower_bound(0)
        .describe("Number of features used in classification.");
    DMLC_DECLARE_FIELD(num_output_group)
        .set_lower_bound(1)
        .set_default(1)
        .describe("Number of output groups in the setting.");
  }
};

// model for linear booster
class GBLinearModel {
 public:
  // parameter
  GBLinearModelParam param;
  // weight for each of feature, bias is the last one
  std::vector<bst_float> weight;
  // initialize the model parameter
  inline void LazyInitModel(void) {
    if (!weight.empty()) return;
    // bias is the last weight
    weight.resize((param.num_feature + 1) * param.num_output_group);
    std::fill(weight.begin(), weight.end(), 0.0f);
  }
  // save the model to file
  inline void Save(dmlc::Stream* fo) const {
    fo->Write(&param, sizeof(param));
    fo->Write(weight);
  }
  // load model from file
  inline void Load(dmlc::Stream* fi) {
    CHECK_EQ(fi->Read(&param, sizeof(param)), sizeof(param));
    fi->Read(&weight);
  }
  // model bias
  inline bst_float* bias() {
    return &weight[param.num_feature * param.num_output_group];
  }
  inline const bst_float* bias() const {
    return &weight[param.num_feature * param.num_output_group];
  }
  // get i-th weight
  inline bst_float* operator[](size_t i) {
    return &weight[i * param.num_output_group];
  }
  inline const bst_float* operator[](size_t i) const {
    return &weight[i * param.num_output_group];
  }
};
}  // namespace gbm
}  // namespace xgboost
