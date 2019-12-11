/*!
 * Copyright 2018-2019 by Contributors
 */
#pragma once
#include <dmlc/io.h>
#include <dmlc/parameter.h>
#include <xgboost/learner.h>

#include <vector>
#include <string>
#include <cstring>

#include "xgboost/base.h"
#include "xgboost/feature_map.h"
#include "xgboost/model.h"
#include "xgboost/json.h"
#include "xgboost/parameter.h"

namespace xgboost {
class Json;
namespace gbm {
// Deprecated in 1.0.0. model parameter.  Only staying here for compatible binary model IO.
struct DeprecatedGBLinearModelParam : public dmlc::Parameter<DeprecatedGBLinearModelParam> {
  // number of feature dimension
  uint32_t deprecated_num_feature;
  // deprecated. use learner_model_param_->num_output_group.
  int32_t deprecated_num_output_group;
  // reserved field
  int32_t reserved[32];
  // constructor
  DeprecatedGBLinearModelParam() {
    static_assert(sizeof(*this) == sizeof(int32_t) * 34,
                  "Model parameter size can not be changed.");
    std::memset(this, 0, sizeof(DeprecatedGBLinearModelParam));
  }

  DMLC_DECLARE_PARAMETER(DeprecatedGBLinearModelParam) {}
};

// model for linear booster
class GBLinearModel : public Model {
 private:
  // Deprecated in 1.0.0
  DeprecatedGBLinearModelParam param;

 public:
  LearnerModelParam const* learner_model_param_;

 public:
  explicit GBLinearModel(LearnerModelParam const* learner_model_param) :
      learner_model_param_ {learner_model_param} {}
  void Configure(Args const &cfg) { }

  // weight for each of feature, bias is the last one
  std::vector<bst_float> weight;
  // initialize the model parameter
  inline void LazyInitModel() {
    if (!weight.empty())
      return;
    // bias is the last weight
    weight.resize((learner_model_param_->num_feature + 1) *
                  learner_model_param_->num_output_group);
    std::fill(weight.begin(), weight.end(), 0.0f);
  }

  void SaveModel(Json *p_out) const override;
  void LoadModel(Json const &in) override;

  // save the model to file
  void Save(dmlc::Stream *fo) const {
    fo->Write(&param, sizeof(param));
    fo->Write(weight);
  }
  // load model from file
  void Load(dmlc::Stream *fi) {
    CHECK_EQ(fi->Read(&param, sizeof(param)), sizeof(param));
    fi->Read(&weight);
  }

  // model bias
  inline bst_float *bias() {
    return &weight[learner_model_param_->num_feature *
                   learner_model_param_->num_output_group];
  }
  inline const bst_float *bias() const {
    return &weight[learner_model_param_->num_feature *
                   learner_model_param_->num_output_group];
  }
  // get i-th weight
  inline bst_float *operator[](size_t i) {
    return &weight[i * learner_model_param_->num_output_group];
  }
  inline const bst_float *operator[](size_t i) const {
    return &weight[i * learner_model_param_->num_output_group];
  }

  std::vector<std::string> DumpModel(const FeatureMap &fmap, bool with_stats,
                                     std::string format) const {
    const int ngroup = learner_model_param_->num_output_group;
    const unsigned nfeature = learner_model_param_->num_feature;

    std::stringstream fo("");
    if (format == "json") {
      fo << "  { \"bias\": [" << std::endl;
      for (int gid = 0; gid < ngroup; ++gid) {
        if (gid != 0)
          fo << "," << std::endl;
        fo << "      " << this->bias()[gid];
      }
      fo << std::endl
         << "    ]," << std::endl
         << "    \"weight\": [" << std::endl;
      for (unsigned i = 0; i < nfeature; ++i) {
        for (int gid = 0; gid < ngroup; ++gid) {
          if (i != 0 || gid != 0)
            fo << "," << std::endl;
          fo << "      " << (*this)[i][gid];
        }
      }
      fo << std::endl << "    ]" << std::endl << "  }";
    } else {
      fo << "bias:\n";
      for (int gid = 0; gid < ngroup; ++gid) {
        fo << this->bias()[gid] << std::endl;
      }
      fo << "weight:\n";
      for (unsigned i = 0; i < nfeature; ++i) {
        for (int gid = 0; gid < ngroup; ++gid) {
          fo << (*this)[i][gid] << std::endl;
        }
      }
    }
    std::vector<std::string> v;
    v.push_back(fo.str());
    return v;
  }
};

}  // namespace gbm
}  // namespace xgboost
