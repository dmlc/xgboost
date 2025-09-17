/**
 * Copyright 2018-2025, XGBoost Contributors
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

namespace xgboost {
class Json;
namespace gbm {
// model for linear booster
class GBLinearModel : public Model {
 public:
  std::int32_t num_boosted_rounds{0};
  LearnerModelParam const* learner_model_param;

 public:
  explicit GBLinearModel(LearnerModelParam const *learner_model_param)
      : learner_model_param{learner_model_param} {}
  void Configure(Args const &) { }

  // weight for each of feature, bias is the last one
  std::vector<bst_float> weight;
  // initialize the model parameter
  inline void LazyInitModel() {
    if (!weight.empty()) {
      return;
    }
    // bias is the last weight
    weight.resize((learner_model_param->num_feature + 1) *
                  learner_model_param->num_output_group);
    std::fill(weight.begin(), weight.end(), 0.0f);
  }

  void SaveModel(Json *p_out) const override;
  void LoadModel(Json const &in) override;

  // model bias
  inline bst_float *Bias() {
    return &weight[learner_model_param->num_feature *
                   learner_model_param->num_output_group];
  }
  inline const bst_float *Bias() const {
    return &weight[learner_model_param->num_feature *
                   learner_model_param->num_output_group];
  }
  // get i-th weight
  inline bst_float *operator[](size_t i) {
    return &weight[i * learner_model_param->num_output_group];
  }
  inline const bst_float *operator[](size_t i) const {
    return &weight[i * learner_model_param->num_output_group];
  }

  std::vector<std::string> DumpModel(const FeatureMap &, bool,
                                     std::string format) const {
    const int ngroup = learner_model_param->num_output_group;
    const unsigned nfeature = learner_model_param->num_feature;

    std::stringstream fo("");
    if (format == "json") {
      fo << "  { \"bias\": [" << std::endl;
      for (int gid = 0; gid < ngroup; ++gid) {
        if (gid != 0) {
          fo << "," << std::endl;
        }
        fo << "      " << this->Bias()[gid];
      }
      fo << std::endl
         << "    ]," << std::endl
         << "    \"weight\": [" << std::endl;
      for (unsigned i = 0; i < nfeature; ++i) {
        for (int gid = 0; gid < ngroup; ++gid) {
          if (i != 0 || gid != 0) {
            fo << "," << std::endl;
          }
          fo << "      " << (*this)[i][gid];
        }
      }
      fo << std::endl << "    ]" << std::endl << "  }";
    } else if (format == "text") {
      fo << "bias:\n";
      for (int gid = 0; gid < ngroup; ++gid) {
        fo << this->Bias()[gid] << std::endl;
      }
      fo << "weight:\n";
      for (unsigned i = 0; i < nfeature; ++i) {
        for (int gid = 0; gid < ngroup; ++gid) {
          fo << (*this)[i][gid] << std::endl;
        }
      }
    } else {
      LOG(FATAL) << "Dump format `" << format << "` is not supported by the gblinear model.";
    }
    std::vector<std::string> v;
    v.push_back(fo.str());
    return v;
  }
};

}  // namespace gbm
}  // namespace xgboost
