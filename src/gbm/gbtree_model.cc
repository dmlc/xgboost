/*!
 * Copyright 2019 by Contributors
 */
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "gbtree_model.h"

namespace xgboost {
namespace gbm {

void GBTreeModel::Load(Json const& in) {
  param.InitAllowUnknown(fromJson(get<Object>(in["model_param"])));

  trees.clear();
  trees_to_update.clear();

  auto const& trees_json = get<Array>(in["trees"]);
  trees.resize(trees_json.size());

  for (size_t t = 0; t < trees.size(); ++t) {
    trees[t].reset( new RegTree() );
    trees[t]->Load(trees_json[t]);
  }

  tree_info.resize(param.num_trees);
}

void GBTreeModel::Save(Json* p_out) const {
  auto& out = *p_out;
  CHECK_EQ(param.num_trees, static_cast<int>(trees.size()));
  out["model_param"] = toJson(param);
  std::vector<Json> trees_json;
  size_t t = 0;
  for (auto const& tree : trees) {
    Json tree_json{Object()};
    tree->Save(&tree_json);
    tree_json["id"] = std::to_string(t);
    trees_json.emplace_back(tree_json);
    t++;
  }

  out["trees"] = Array(trees_json);
}

}  // namespace gbm
}  // namespace xgboost
