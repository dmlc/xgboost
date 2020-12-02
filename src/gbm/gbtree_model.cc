/*!
 * Copyright 2019-2020 by Contributors
 */
#include <utility>

#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "gbtree_model.h"
#include "gbtree.h"

namespace xgboost {
namespace gbm {
void GBTreeModel::Save(dmlc::Stream* fo) const {
  CHECK_EQ(param.num_trees, static_cast<int32_t>(trees.size()));

  if (DMLC_IO_NO_ENDIAN_SWAP) {
    fo->Write(&param, sizeof(param));
  } else {
    auto x = param.ByteSwap();
    fo->Write(&x, sizeof(x));
  }
  for (const auto & tree : trees) {
    tree->Save(fo);
  }
  if (tree_info.size() != 0) {
    if (DMLC_IO_NO_ENDIAN_SWAP) {
      fo->Write(dmlc::BeginPtr(tree_info), sizeof(int32_t) * tree_info.size());
    } else {
      for (const auto& e : tree_info) {
        auto x = e;
        dmlc::ByteSwap(&x, sizeof(x), 1);
        fo->Write(&x, sizeof(x));
      }
    }
  }
}

void GBTreeModel::Load(dmlc::Stream* fi) {
  CHECK_EQ(fi->Read(&param, sizeof(param)), sizeof(param))
      << "GBTree: invalid model file";
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    param = param.ByteSwap();
  }
  trees.clear();
  trees_to_update.clear();
  for (int32_t i = 0; i < param.num_trees; ++i) {
    std::unique_ptr<RegTree> ptr(new RegTree());
    ptr->Load(fi);
    trees.push_back(std::move(ptr));
  }
  tree_info.resize(param.num_trees);
  if (param.num_trees != 0) {
    if (DMLC_IO_NO_ENDIAN_SWAP) {
      CHECK_EQ(
          fi->Read(dmlc::BeginPtr(tree_info), sizeof(int32_t) * param.num_trees),
          sizeof(int32_t) * param.num_trees);
    } else {
      for (auto& info : tree_info) {
        CHECK_EQ(fi->Read(&info, sizeof(int32_t)), sizeof(int32_t));
        dmlc::ByteSwap(&info, sizeof(info), 1);
      }
    }
  }
}

void GBTreeModel::SaveModel(Json* p_out) const {
  auto& out = *p_out;
  CHECK_EQ(param.num_trees, static_cast<int>(trees.size()));
  out["gbtree_model_param"] = ToJson(param);
  std::vector<Json> trees_json(trees.size());

  for (size_t t = 0; t < trees.size(); ++t) {
    auto const& tree = trees[t];
    Json tree_json{Object()};
    tree->SaveModel(&tree_json);
    tree_json["id"] = Integer(static_cast<Integer::Int>(t));
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

  for (size_t t = 0; t < trees_json.size(); ++t) {  // NOLINT
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
