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
void DumpRegTree(std::stringstream& fo,  // NOLINT(*)
                 const RegTree& tree,
                 const FeatureMap& fmap,
                 int nid, int depth, int add_comma,
                 bool with_stats, std::string format) {
  if (format == "json") {
    if (add_comma) fo << ",";
    if (depth != 0) fo << std::endl;
    for (int i = 0; i < depth+1; ++i) fo << "  ";
  } else {
    for (int i = 0; i < depth; ++i) fo << '\t';
  }
  if (tree[nid].is_leaf()) {
    if (format == "json") {
      fo << "{ \"nodeid\": " << nid
         << ", \"leaf\": " << tree[nid].leaf_value();
      if (with_stats) {
        fo << ", \"cover\": " << tree.stat(nid).sum_hess;
      }
      fo << " }";
    } else {
      fo << nid << ":leaf=" << tree[nid].leaf_value();
      if (with_stats) {
        fo << ",cover=" << tree.stat(nid).sum_hess;
      }
      fo << '\n';
    }
  } else {
    // right then left,
    bst_float cond = tree[nid].split_cond();
    const unsigned split_index = tree[nid].split_index();
    if (split_index < fmap.size()) {
      switch (fmap.type(split_index)) {
        case FeatureMap::kIndicator: {
          int nyes = tree[nid].default_left() ?
              tree[nid].cright() : tree[nid].cleft();
          if (format == "json") {
            fo << "{ \"nodeid\": " << nid
               << ", \"depth\": " << depth
               << ", \"split\": \"" << fmap.name(split_index) << "\""
               << ", \"yes\": " << nyes
               << ", \"no\": " << tree[nid].cdefault();
          } else {
            fo << nid << ":[" << fmap.name(split_index) << "] yes=" << nyes
               << ",no=" << tree[nid].cdefault();
          }
          break;
        }
        case FeatureMap::kInteger: {
          if (format == "json") {
            fo << "{ \"nodeid\": " << nid
               << ", \"depth\": " << depth
               << ", \"split\": \"" << fmap.name(split_index) << "\""
               << ", \"split_condition\": " << int(cond + 1.0)
               << ", \"yes\": " << tree[nid].cleft()
               << ", \"no\": " << tree[nid].cright()
               << ", \"missing\": " << tree[nid].cdefault();
          } else {
            fo << nid << ":[" << fmap.name(split_index) << "<"
               << int(cond + 1.0)
               << "] yes=" << tree[nid].cleft()
               << ",no=" << tree[nid].cright()
               << ",missing=" << tree[nid].cdefault();
          }
          break;
        }
        case FeatureMap::kFloat:
        case FeatureMap::kQuantitive: {
          if (format == "json") {
            fo << "{ \"nodeid\": " << nid
               << ", \"depth\": " << depth
               << ", \"split\": \"" << fmap.name(split_index) << "\""
               << ", \"split_condition\": " << cond
               << ", \"yes\": " << tree[nid].cleft()
               << ", \"no\": " << tree[nid].cright()
               << ", \"missing\": " << tree[nid].cdefault();
          } else {
            fo << nid << ":[" << fmap.name(split_index) << "<" << cond
               << "] yes=" << tree[nid].cleft()
               << ",no=" << tree[nid].cright()
               << ",missing=" << tree[nid].cdefault();
          }
          break;
        }
        default: LOG(FATAL) << "unknown fmap type";
        }
    } else {
      if (format == "json") {
        fo << "{ \"nodeid\": " << nid
           << ", \"depth\": " << depth
           << ", \"split\": " << split_index
           << ", \"split_condition\": " << cond
           << ", \"yes\": " << tree[nid].cleft()
           << ", \"no\": " << tree[nid].cright()
           << ", \"missing\": " << tree[nid].cdefault();
      } else {
        fo << nid << ":[f" << split_index << "<"<< cond
           << "] yes=" << tree[nid].cleft()
           << ",no=" << tree[nid].cright()
           << ",missing=" << tree[nid].cdefault();
      }
    }
    if (with_stats) {
      if (format == "json") {
        fo << ", \"gain\": " << tree.stat(nid).loss_chg
           << ", \"cover\": " << tree.stat(nid).sum_hess;
      } else {
        fo << ",gain=" << tree.stat(nid).loss_chg << ",cover=" << tree.stat(nid).sum_hess;
      }
    }
    if (format == "json") {
      fo << ", \"children\": [";
    } else {
      fo << '\n';
    }
    DumpRegTree(fo, tree, fmap, tree[nid].cleft(), depth + 1, false, with_stats, format);
    DumpRegTree(fo, tree, fmap, tree[nid].cright(), depth + 1, true, with_stats, format);
    if (format == "json") {
      fo << std::endl;
      for (int i = 0; i < depth+1; ++i) fo << "  ";
      fo << "]}";
    }
  }
}

std::string RegTree::DumpModel(const FeatureMap& fmap,
                               bool with_stats,
                               std::string format) const {
  std::stringstream fo("");
  for (int i = 0; i < param.num_roots; ++i) {
    DumpRegTree(fo, *this, fmap, i, 0, false, with_stats, format);
  }
  return fo.str();
}
}  // namespace xgboost
