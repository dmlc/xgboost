/*!
 * Copyright by Contributors 2018
 */
#pragma once
#include <dmlc/io.h>
#include <dmlc/parameter.h>
#include <xgboost/feature_map.h>
#include <vector>
#include <string>
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
  inline void LazyInitModel() {
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

  std::vector<std::string> DumpModel(const FeatureMap& fmap, bool with_stats,
                                     std::string format) const {
    const int ngroup = param.num_output_group;
    const unsigned nfeature = param.num_feature;

    std::stringstream fo("");
    if (format == "json") {
      fo << "  { \"bias\": [" << std::endl;
      for (int gid = 0; gid < ngroup; ++gid) {
        if (gid != 0) fo << "," << std::endl;
        fo << "      " << this->bias()[gid];
      }
      fo << std::endl << "    ]," << std::endl
         << "    \"weight\": [" << std::endl;
      for (unsigned i = 0; i < nfeature; ++i) {
        for (int gid = 0; gid < ngroup; ++gid) {
          if (i != 0 || gid != 0) fo << "," << std::endl;
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
