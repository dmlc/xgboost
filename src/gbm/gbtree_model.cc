/*!
 * Copyright 2019 by Contributors
 */
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "gbtree_model.h"
#include <utility>

namespace xgboost {
namespace gbm {

void GBTreeModel::Save(dmlc::Stream* fo) const {
  CHECK_EQ(param.num_trees, static_cast<int32_t>(trees.size()));
  fo->Write(&param, sizeof(param));
  for (const auto & tree : trees) {
    tree->Save(fo);
  }
  if (tree_info.size() != 0) {
    fo->Write(dmlc::BeginPtr(tree_info), sizeof(int32_t) * tree_info.size());
  }
}

void GBTreeModel::Load(dmlc::Stream* fi) {
  CHECK_EQ(fi->Read(&param, sizeof(param)), sizeof(param))
      << "GBTree: invalid model file";
  trees.clear();
  trees_to_update.clear();
  for (int32_t i = 0; i < param.num_trees; ++i) {
    std::unique_ptr<RegTree> ptr(new RegTree());
    ptr->Load(fi);
    trees.push_back(std::move(ptr));
  }
  tree_info.resize(param.num_trees);
  if (param.num_trees != 0) {
    CHECK_EQ(
        fi->Read(dmlc::BeginPtr(tree_info), sizeof(int32_t) * param.num_trees),
        sizeof(int32_t) * param.num_trees);
  }
}

void GBTreeModel::SaveModel(Json* p_out) const {
  auto& out = *p_out;
  CHECK_EQ(param.num_trees, static_cast<int>(trees.size()));
  out["gbtree_model_param"] = ToJson(param);
  std::vector<Json> trees_json(trees.size());

#pragma omp parallel for
  for (omp_ulong t = 0; t < trees.size(); ++t) {
    auto const& tree = trees[t];
    Json tree_json{Object()};
    tree->SaveModel(&tree_json);
    tree_json["id"] = Integer(t);
    trees_json[t] = std::move(tree_json);
  }

  std::vector<Json> tree_info_json(tree_info.size());
  for (size_t i = 0; i < tree_info.size(); ++i) {
    tree_info_json[i] = Integer(tree_info[i]);
  }

  out["trees"] = Array(std::move(trees_json));
  out["tree_info"] = Array(std::move(tree_info_json));
}

void GBTreeModel::LoadModel(Json const& in) {
  FromJson(in["gbtree_model_param"], &param);

  trees.clear();
  trees_to_update.clear();

  auto const& trees_json = get<Array const>(in["trees"]);
  trees.resize(trees_json.size());

#pragma omp parallel for
  for (omp_ulong t = 0; t < trees_json.size(); ++t) {  // NOLINT
    auto tree_id = get<Integer>(trees_json[t]["id"]);
    trees.at(tree_id).reset(new RegTree());
    trees.at(tree_id)->LoadModel(trees_json[t]);
  }

  tree_info.resize(param.num_trees);
  auto const& tree_info_json = get<Array const>(in["tree_info"]);
  for (int32_t i = 0; i < param.num_trees; ++i) {
    tree_info[i] = get<Integer const>(tree_info_json[i]);
  }
}

}  // namespace gbm
}  // namespace xgboost
