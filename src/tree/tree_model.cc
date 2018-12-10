/*!
 * Copyright 2015 by Contributors
 * \file tree_model.cc
 * \brief model structure for tree
 */
#include <xgboost/tree_model.h>
#include <sstream>
#include <limits>
#include <cmath>
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
  const auto &node = tree.GetNode(nid);
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
  if (node.IsLeaf()) {
    if (format == "json") {
      fo << "{ \"nodeid\": " << nid
         << ", \"leaf\": " << std::setprecision(float_max_precision) << node.LeafValue();
      if (with_stats) {
        fo << ", \"cover\": " << std::setprecision(float_max_precision) << tree.Stat(nid).sum_hess;
      }
      fo << " }";
    } else {
      fo << nid << ":leaf=" << std::setprecision(float_max_precision) << node.LeafValue();
      if (with_stats) {
        fo << ",cover=" << std::setprecision(float_max_precision) << tree.Stat(nid).sum_hess;
      }
      fo << '\n';
    }
  } else {
    // right then left,
    bst_float cond = node.SplitCond();
    const unsigned split_index = node.SplitIndex();
    if (split_index < fmap.Size()) {
      switch (fmap.type(split_index)) {
        case FeatureMap::kIndicator: {
          int nyes = node.DefaultLeft() ?
              node.RightChild() : node.LeftChild();
          if (format == "json") {
            fo << "{ \"nodeid\": " << nid
               << ", \"depth\": " << depth
               << ", \"split\": \"" << fmap.Name(split_index) << "\""
               << ", \"yes\": " << nyes
               << ", \"no\": " << node.DefaultChild();
          } else {
            fo << nid << ":[" << fmap.Name(split_index) << "] yes=" << nyes
               << ",no=" << node.DefaultChild();
          }
          break;
        }
        case FeatureMap::kInteger: {
          const bst_float floored = std::floor(cond);
          const int integer_threshold
            = (floored == cond) ? static_cast<int>(floored)
                                : static_cast<int>(floored) + 1;
          if (format == "json") {
            fo << "{ \"nodeid\": " << nid
               << ", \"depth\": " << depth
               << ", \"split\": \"" << fmap.Name(split_index) << "\""
               << ", \"split_condition\": " << integer_threshold
               << ", \"yes\": " << node.LeftChild()
               << ", \"no\": " << node.RightChild()
               << ", \"missing\": " << node.DefaultChild();
          } else {
            fo << nid << ":[" << fmap.Name(split_index) << "<"
               << integer_threshold
               << "] yes=" << node.LeftChild()
               << ",no=" << node.RightChild()
               << ",missing=" << node.DefaultChild();
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
               << ", \"yes\": " << node.LeftChild()
               << ", \"no\": " << node.RightChild()
               << ", \"missing\": " << node.DefaultChild();
          } else {
            fo << nid << ":[" << fmap.Name(split_index)
               << "<" << std::setprecision(float_max_precision) << cond
               << "] yes=" << node.LeftChild()
               << ",no=" << node.RightChild()
               << ",missing=" << node.DefaultChild();
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
           << ", \"yes\": " << node.LeftChild()
           << ", \"no\": " << node.RightChild()
           << ", \"missing\": " << node.DefaultChild();
      } else {
        fo << nid << ":[f" << split_index << "<"<< std::setprecision(float_max_precision) << cond
           << "] yes=" << node.LeftChild()
           << ",no=" << node.RightChild()
           << ",missing=" << node.DefaultChild();
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
    DumpRegTree(fo, tree, fmap, node.LeftChild(), depth + 1, false, with_stats, format);
    DumpRegTree(fo, tree, fmap, node.RightChild(), depth + 1, true, with_stats, format);
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
  DumpRegTree(fo, *this, fmap, 0, 0, false, with_stats, format);
  return fo.str();
}
}  // namespace xgboost
