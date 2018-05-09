/*!
 * Copyright 2015 by Contributors
 * \file tree_model.cc
 * \brief model structure for tree
 */
#include <xgboost/tree_model.h>
#include <sstream>
#include <limits>
#include <iomanip>
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
  int float_max_precision = std::numeric_limits<bst_float>::max_digits10;
  if (format == "json") {
    if (add_comma) {
      fo << ",";
    }
    if (depth != 0) {
      fo << std::endl;
    }
    for (int i = 0; i < depth + 1; ++i) {
      fo << "  ";
    }
  } else {
    for (int i = 0; i < depth; ++i) {
      fo << '\t';
    }
  }
  if (tree[nid].IsLeaf()) {
    if (format == "json") {
      fo << "{ \"nodeid\": " << nid
         << ", \"leaf\": " << std::setprecision(float_max_precision) << tree[nid].LeafValue();
      if (with_stats) {
        fo << ", \"cover\": " << std::setprecision(float_max_precision) << tree.Stat(nid).sum_hess;
      }
      fo << " }";
    } else {
      fo << nid << ":leaf=" << std::setprecision(float_max_precision) << tree[nid].LeafValue();
      if (with_stats) {
        fo << ",cover=" << std::setprecision(float_max_precision) << tree.Stat(nid).sum_hess;
      }
      fo << '\n';
    }
  } else {
    // right then left,
    bst_float cond = tree[nid].SplitCond();
    const unsigned split_index = tree[nid].SplitIndex();
    if (split_index < fmap.Size()) {
      switch (fmap.type(split_index)) {
        case FeatureMap::kIndicator: {
          int nyes = tree[nid].DefaultLeft() ?
              tree[nid].RightChild() : tree[nid].LeftChild();
          if (format == "json") {
            fo << "{ \"nodeid\": " << nid
               << ", \"depth\": " << depth
               << ", \"split\": \"" << fmap.Name(split_index) << "\""
               << ", \"yes\": " << nyes
               << ", \"no\": " << tree[nid].DefaultChild();
          } else {
            fo << nid << ":[" << fmap.Name(split_index) << "] yes=" << nyes
               << ",no=" << tree[nid].DefaultChild();
          }
          break;
        }
        case FeatureMap::kInteger: {
          if (format == "json") {
            fo << "{ \"nodeid\": " << nid
               << ", \"depth\": " << depth
               << ", \"split\": \"" << fmap.Name(split_index) << "\""
               << ", \"split_condition\": " << int(cond + 1.0)
               << ", \"yes\": " << tree[nid].LeftChild()
               << ", \"no\": " << tree[nid].RightChild()
               << ", \"missing\": " << tree[nid].DefaultChild();
          } else {
            fo << nid << ":[" << fmap.Name(split_index) << "<"
               << int(cond + 1.0)
               << "] yes=" << tree[nid].LeftChild()
               << ",no=" << tree[nid].RightChild()
               << ",missing=" << tree[nid].DefaultChild();
          }
          break;
        }
        case FeatureMap::kFloat:
        case FeatureMap::kQuantitive: {
          if (format == "json") {
            fo << "{ \"nodeid\": " << nid
               << ", \"depth\": " << depth
               << ", \"split\": \"" << fmap.Name(split_index) << "\""
               << ", \"split_condition\": " << std::setprecision(float_max_precision) << cond
               << ", \"yes\": " << tree[nid].LeftChild()
               << ", \"no\": " << tree[nid].RightChild()
               << ", \"missing\": " << tree[nid].DefaultChild();
          } else {
            fo << nid << ":[" << fmap.Name(split_index)
               << "<" << std::setprecision(float_max_precision) << cond
               << "] yes=" << tree[nid].LeftChild()
               << ",no=" << tree[nid].RightChild()
               << ",missing=" << tree[nid].DefaultChild();
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
           << ", \"split_condition\": " << std::setprecision(float_max_precision) << cond
           << ", \"yes\": " << tree[nid].LeftChild()
           << ", \"no\": " << tree[nid].RightChild()
           << ", \"missing\": " << tree[nid].DefaultChild();
      } else {
        fo << nid << ":[f" << split_index << "<"<< std::setprecision(float_max_precision) << cond
           << "] yes=" << tree[nid].LeftChild()
           << ",no=" << tree[nid].RightChild()
           << ",missing=" << tree[nid].DefaultChild();
      }
    }
    if (with_stats) {
      if (format == "json") {
        fo << ", \"gain\": " << std::setprecision(float_max_precision) << tree.Stat(nid).loss_chg
           << ", \"cover\": " << std::setprecision(float_max_precision) << tree.Stat(nid).sum_hess;
      } else {
        fo << ",gain=" << std::setprecision(float_max_precision) << tree.Stat(nid).loss_chg
           << ",cover=" << std::setprecision(float_max_precision) << tree.Stat(nid).sum_hess;
      }
    }
    if (format == "json") {
      fo << ", \"children\": [";
    } else {
      fo << '\n';
    }
    DumpRegTree(fo, tree, fmap, tree[nid].LeftChild(), depth + 1, false, with_stats, format);
    DumpRegTree(fo, tree, fmap, tree[nid].RightChild(), depth + 1, true, with_stats, format);
    if (format == "json") {
      fo << std::endl;
      for (int i = 0; i < depth + 1; ++i) {
        fo << "  ";
      }
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
