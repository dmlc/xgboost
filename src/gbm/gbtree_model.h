/*!
 * Copyright 2017-2020 by Contributors
 * \file gbtree_model.h
 */
#ifndef XGBOOST_GBM_GBTREE_MODEL_H_
#define XGBOOST_GBM_GBTREE_MODEL_H_

#include <dmlc/parameter.h>
#include <dmlc/io.h>
#include <xgboost/model.h>
#include <xgboost/tree_model.h>
#include <xgboost/parameter.h>
#include <xgboost/learner.h>

#include <memory>
#include <utility>
#include <string>
#include <vector>

namespace xgboost {

class Json;

namespace gbm {

/*! \brief model parameters */
struct GBTreeModelParam : public dmlc::Parameter<GBTreeModelParam> {
 public:
  /*! \brief number of trees */
  int32_t num_trees;
  /*! \brief (Deprecated) number of roots */
  int32_t deprecated_num_roots;
  /*! \brief number of features to be used by trees */
  int32_t deprecated_num_feature;
  /*! \brief pad this space, for backward compatibility reason.*/
  int32_t pad_32bit;
  /*! \brief deprecated padding space. */
  int64_t deprecated_num_pbuffer;
  // deprecated. use learner_model_param_->num_output_group.
  int32_t deprecated_num_output_group;
  /*! \brief size of leaf vector needed in tree */
  int32_t size_leaf_vector;
  /*! \brief reserved parameters */
  int32_t reserved[32];

  /*! \brief constructor */
  GBTreeModelParam() {
    std::memset(this, 0, sizeof(GBTreeModelParam));  // FIXME(trivialfis): Why?
    static_assert(sizeof(GBTreeModelParam) == (4 + 2 + 2 + 32) * sizeof(int32_t),
                  "64/32 bit compatibility issue");
    deprecated_num_roots = 1;
  }

  // declare parameters, only declare those that need to be set.
  DMLC_DECLARE_PARAMETER(GBTreeModelParam) {
    DMLC_DECLARE_FIELD(num_trees)
        .set_lower_bound(0)
        .set_default(0)
        .describe("Number of features used for training and prediction.");
    DMLC_DECLARE_FIELD(size_leaf_vector)
        .set_lower_bound(0)
        .set_default(0)
        .describe("Reserved option for vector tree.");
  }

  // Swap byte order for all fields. Useful for transporting models between machines with different
  // endianness (big endian vs little endian)
  inline GBTreeModelParam ByteSwap() const {
    GBTreeModelParam x = *this;
    dmlc::ByteSwap(&x.num_trees, sizeof(x.num_trees), 1);
    dmlc::ByteSwap(&x.deprecated_num_roots, sizeof(x.deprecated_num_roots), 1);
    dmlc::ByteSwap(&x.deprecated_num_feature, sizeof(x.deprecated_num_feature), 1);
    dmlc::ByteSwap(&x.pad_32bit, sizeof(x.pad_32bit), 1);
    dmlc::ByteSwap(&x.deprecated_num_pbuffer, sizeof(x.deprecated_num_pbuffer), 1);
    dmlc::ByteSwap(&x.deprecated_num_output_group, sizeof(x.deprecated_num_output_group), 1);
    dmlc::ByteSwap(&x.size_leaf_vector, sizeof(x.size_leaf_vector), 1);
    dmlc::ByteSwap(x.reserved, sizeof(x.reserved[0]), sizeof(x.reserved) / sizeof(x.reserved[0]));
    return x;
  }
};

struct GBTreeModel : public Model {
 public:
  explicit GBTreeModel(LearnerModelParam const* learner_model) :
      learner_model_param{learner_model} {}
  void Configure(const Args& cfg) {
    // initialize model parameters if not yet been initialized.
    if (trees.size() == 0) {
      param.UpdateAllowUnknown(cfg);
    }
  }

  void InitTreesToUpdate() {
    if (trees_to_update.size() == 0u) {
      for (auto & tree : trees) {
        trees_to_update.push_back(std::move(tree));
      }
      trees.clear();
      param.num_trees = 0;
      tree_info.clear();
    }
  }

  void Load(dmlc::Stream* fi);
  void Save(dmlc::Stream* fo) const;

  void SaveModel(Json* p_out) const override;
  void LoadModel(Json const& p_out) override;

  std::vector<std::string> DumpModel(const FeatureMap& fmap, bool with_stats,
                                     std::string format) const {
    std::vector<std::string> dump;
    for (const auto & tree : trees) {
      dump.push_back(tree->DumpModel(fmap, with_stats, format));
    }
    return dump;
  }
  void CommitModel(std::vector<std::unique_ptr<RegTree> >&& new_trees,
                   int bst_group) {
    for (auto & new_tree : new_trees) {
      trees.push_back(std::move(new_tree));
      tree_info.push_back(bst_group);
    }
    param.num_trees += static_cast<int>(new_trees.size());
  }

  // base margin
  LearnerModelParam const* learner_model_param;
  // model parameter
  GBTreeModelParam param;
  /*! \brief vector of trees stored in the model */
  std::vector<std::unique_ptr<RegTree> > trees;
  /*! \brief for the update process, a place to keep the initial trees */
  std::vector<std::unique_ptr<RegTree> > trees_to_update;
  /*! \brief some information indicator of the tree, reserved */
  std::vector<int> tree_info;
};
}  // namespace gbm
}  // namespace xgboost

#endif  // XGBOOST_GBM_GBTREE_MODEL_H_
