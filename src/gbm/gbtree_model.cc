/**
 * Copyright 2019-2023, XGBoost Contributors
 */
#include "gbtree_model.h"

#include <cstddef>                      // for size_t
#include <ostream>                      // for operator<<, basic_ostream
#include <utility>                      // for move

#include "../common/threading_utils.h"  // for ParallelFor
#include "dmlc/base.h"                  // for BeginPtr
#include "dmlc/io.h"                    // for Stream
#include "xgboost/context.h"            // for Context
#include "xgboost/json.h"               // for Json, get, Integer, Array, FromJson, ToJson, Object
#include "xgboost/logging.h"            // for LogCheck_EQ, CHECK_EQ, CHECK
#include "xgboost/tree_model.h"         // for RegTree

namespace xgboost::gbm {
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

  CHECK(ctx_);
  common::ParallelFor(trees.size(), ctx_->Threads(), [&](auto t) {
    auto const& tree = trees[t];
    Json jtree{Object{}};
    tree->SaveModel(&jtree);
    jtree["id"] = Integer{static_cast<Integer::Int>(t)};
    trees_json[t] = std::move(jtree);
  });

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
  CHECK_EQ(trees_json.size(), param.num_trees);
  trees.resize(param.num_trees);

  auto const& tree_info_json = get<Array const>(in["tree_info"]);
  CHECK_EQ(tree_info_json.size(), param.num_trees);
  tree_info.resize(param.num_trees);

  CHECK(ctx_);

  common::ParallelFor(param.num_trees, ctx_->Threads(), [&](auto t) {
    auto tree_id = get<Integer const>(trees_json[t]["id"]);
    trees.at(tree_id).reset(new RegTree{});
    trees[tree_id]->LoadModel(trees_json[t]);
  });

  for (bst_tree_t i = 0; i < param.num_trees; ++i) {
    tree_info[i] = get<Integer const>(tree_info_json[i]);
  }
}

std::uint32_t GBTreeModel::CommitModel(TreesOneIter&& new_trees) {
  CHECK(!iteration_indptr.empty());
  CHECK_EQ(iteration_indptr.back(), param.num_trees);
  std::uint32_t n_new_trees{0};

  if (learner_model_param->IsVectorLeaf()) {
    n_new_trees += new_trees.front().size();
    this->CommitModelGroup(std::move(new_trees.front()), 0);
  } else {
    for (bst_target_t gidx{0}; gidx < learner_model_param->OutputLength(); ++gidx) {
      n_new_trees += new_trees[gidx].size();
      this->CommitModelGroup(std::move(new_trees[gidx]), gidx);
    }
  }

  iteration_indptr.push_back(n_new_trees + iteration_indptr.back());
  return n_new_trees;
}
}  // namespace xgboost::gbm
