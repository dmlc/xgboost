/*!
 * Copyright 2015 by Contributors
 * \file tree_model.cc
 * \brief model structure for tree
 */
#include <xgboost/tree_model.h>
#include <sstream>
#include "./param.h"

namespace xgboost {
// register tree parameter
DMLC_REGISTER_PARAMETER(TreeParam);

namespace tree {
DMLC_REGISTER_PARAMETER(TrainParam);
}
// internal function to dump regression tree to text
void DumpRegTree2Text(std::stringstream& fo,  // NOLINT(*)
                      const RegTree& tree,
                      const FeatureMap& fmap,
                      int nid, int depth, bool with_stats) {
  for (int i = 0;  i < depth; ++i) {
    fo << '\t';
  }
  if (tree[nid].is_leaf()) {
    fo << nid << ":leaf=" << tree[nid].leaf_value();
    if (with_stats) {
      fo << ",cover=" << tree.stat(nid).sum_hess;
    }
    fo << '\n';
  } else {
    // right then left,
    bst_float cond = tree[nid].split_cond();
    const unsigned split_index = tree[nid].split_index();
    if (split_index < fmap.size()) {
      switch (fmap.type(split_index)) {
        case FeatureMap::kIndicator: {
          int nyes = tree[nid].default_left() ?
              tree[nid].cright() : tree[nid].cleft();
          fo << nid << ":[" << fmap.name(split_index) << "] yes=" << nyes
             << ",no=" << tree[nid].cdefault();
          break;
        }
        case FeatureMap::kInteger: {
          fo << nid << ":[" << fmap.name(split_index) << "<"
             // << int(float(cond)+1.0f)
             << float(cond)
             << "] yes=" << tree[nid].cleft()
             << ",no=" << tree[nid].cright()
             << ",missing=" << tree[nid].cdefault();
          break;
        }
        case FeatureMap::kFloat:
        case FeatureMap::kQuantitive: {
          fo << nid << ":[" << fmap.name(split_index) << "<"<< float(cond)
             << "] yes=" << tree[nid].cleft()
             << ",no=" << tree[nid].cright()
             << ",missing=" << tree[nid].cdefault();
            break;
        }
        default: LOG(FATAL) << "unknown fmap type";
        }
    } else {
      fo << nid << ":[f" << split_index << "<"<< float(cond)
         << "] yes=" << tree[nid].cleft()
         << ",no=" << tree[nid].cright()
         << ",missing=" << tree[nid].cdefault();
    }
    if (with_stats) {
      fo << ",gain=" << tree.stat(nid).loss_chg << ",cover=" << tree.stat(nid).sum_hess;
    }
    fo << '\n';
    DumpRegTree2Text(fo, tree, fmap, tree[nid].cleft(), depth + 1, with_stats);
    DumpRegTree2Text(fo, tree, fmap, tree[nid].cright(), depth + 1, with_stats);
  }
}

std::string RegTree::Dump2Text(const FeatureMap& fmap, bool with_stats) const {
  std::stringstream fo("");
  for (int i = 0; i < param.num_roots; ++i) {
    DumpRegTree2Text(fo, *this, fmap, i, 0, with_stats);
  }
  return fo.str();
}
}  // namespace xgboost
