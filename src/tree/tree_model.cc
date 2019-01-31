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
          const bst_float floored = std::floor(cond);
          const int integer_threshold
            = (floored == cond) ? static_cast<int>(floored)
                                : static_cast<int>(floored) + 1;
          if (format == "json") {
            fo << "{ \"nodeid\": " << nid
               << ", \"depth\": " << depth
               << ", \"split\": \"" << fmap.Name(split_index) << "\""
               << ", \"split_condition\": " << integer_threshold
               << ", \"yes\": " << tree[nid].LeftChild()
               << ", \"no\": " << tree[nid].RightChild()
               << ", \"missing\": " << tree[nid].DefaultChild();
          } else {
            fo << nid << ":[" << fmap.Name(split_index) << "<"
               << integer_threshold
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
void RegTree::FillNodeMeanValues() {
  size_t num_nodes = this->param.num_nodes;
  if (this->node_mean_values_.size() == num_nodes) {
    return;
  }
  this->node_mean_values_.resize(num_nodes);
  for (int root_id = 0; root_id < param.num_roots; ++root_id) {
    this->FillNodeMeanValue(root_id);
  }
}

bst_float RegTree::FillNodeMeanValue(int nid) {
  bst_float result;
  auto& node = (*this)[nid];
  if (node.IsLeaf()) {
    result = node.LeafValue();
  } else {
    result  = this->FillNodeMeanValue(node.LeftChild()) * this->Stat(node.LeftChild()).sum_hess;
    result += this->FillNodeMeanValue(node.RightChild()) * this->Stat(node.RightChild()).sum_hess;
    result /= this->Stat(nid).sum_hess;
  }
  this->node_mean_values_[nid] = result;
  return result;
}

void RegTree::CalculateContributionsApprox(const RegTree::FVec &feat,
                                           unsigned root_id,
                                           bst_float *out_contribs) const {
  CHECK_GT(this->node_mean_values_.size(), 0U);
  // this follows the idea of http://blog.datadive.net/interpreting-random-forests/
  unsigned split_index = 0;
  auto pid = static_cast<int>(root_id);
  // update bias value
  bst_float node_value = this->node_mean_values_[pid];
  out_contribs[feat.Size()] += node_value;
  if ((*this)[pid].IsLeaf()) {
    // nothing to do anymore
    return;
  }
  while (!(*this)[pid].IsLeaf()) {
    split_index = (*this)[pid].SplitIndex();
    pid = this->GetNext(pid, feat.Fvalue(split_index), feat.IsMissing(split_index));
    bst_float new_value = this->node_mean_values_[pid];
    // update feature weight
    out_contribs[split_index] += new_value - node_value;
    node_value = new_value;
  }
  bst_float leaf_value = (*this)[pid].LeafValue();
  // update leaf feature weight
  out_contribs[split_index] += leaf_value - node_value;
}

// Used by TreeShap
// data we keep about our decision path
// note that pweight is included for convenience and is not tied with the other attributes
// the pweight of the i'th path element is the permuation weight of paths with i-1 ones in them
struct PathElement {
  int feature_index;
  bst_float zero_fraction;
  bst_float one_fraction;
  bst_float pweight;
  PathElement() = default;
  PathElement(int i, bst_float z, bst_float o, bst_float w) :
    feature_index(i), zero_fraction(z), one_fraction(o), pweight(w) {}
};

// extend our decision path with a fraction of one and zero extensions
void ExtendPath(PathElement *unique_path, unsigned unique_depth,
                bst_float zero_fraction, bst_float one_fraction,
                int feature_index) {
  unique_path[unique_depth].feature_index = feature_index;
  unique_path[unique_depth].zero_fraction = zero_fraction;
  unique_path[unique_depth].one_fraction = one_fraction;
  unique_path[unique_depth].pweight = (unique_depth == 0 ? 1.0f : 0.0f);
  for (int i = unique_depth - 1; i >= 0; i--) {
    unique_path[i+1].pweight += one_fraction * unique_path[i].pweight * (i + 1)
                                / static_cast<bst_float>(unique_depth + 1);
    unique_path[i].pweight = zero_fraction * unique_path[i].pweight * (unique_depth - i)
                             / static_cast<bst_float>(unique_depth + 1);
  }
}

// undo a previous extension of the decision path
void UnwindPath(PathElement *unique_path, unsigned unique_depth,
                unsigned path_index) {
  const bst_float one_fraction = unique_path[path_index].one_fraction;
  const bst_float zero_fraction = unique_path[path_index].zero_fraction;
  bst_float next_one_portion = unique_path[unique_depth].pweight;

  for (int i = unique_depth - 1; i >= 0; --i) {
    if (one_fraction != 0) {
      const bst_float tmp = unique_path[i].pweight;
      unique_path[i].pweight = next_one_portion * (unique_depth + 1)
                               / static_cast<bst_float>((i + 1) * one_fraction);
      next_one_portion = tmp - unique_path[i].pweight * zero_fraction * (unique_depth - i)
                               / static_cast<bst_float>(unique_depth + 1);
    } else {
      unique_path[i].pweight = (unique_path[i].pweight * (unique_depth + 1))
                               / static_cast<bst_float>(zero_fraction * (unique_depth - i));
    }
  }

  for (auto i = path_index; i < unique_depth; ++i) {
    unique_path[i].feature_index = unique_path[i+1].feature_index;
    unique_path[i].zero_fraction = unique_path[i+1].zero_fraction;
    unique_path[i].one_fraction = unique_path[i+1].one_fraction;
  }
}

// determine what the total permuation weight would be if
// we unwound a previous extension in the decision path
bst_float UnwoundPathSum(const PathElement *unique_path, unsigned unique_depth,
                         unsigned path_index) {
  const bst_float one_fraction = unique_path[path_index].one_fraction;
  const bst_float zero_fraction = unique_path[path_index].zero_fraction;
  bst_float next_one_portion = unique_path[unique_depth].pweight;
  bst_float total = 0;
  for (int i = unique_depth - 1; i >= 0; --i) {
    if (one_fraction != 0) {
      const bst_float tmp = next_one_portion * (unique_depth + 1)
                            / static_cast<bst_float>((i + 1) * one_fraction);
      total += tmp;
      next_one_portion = unique_path[i].pweight - tmp * zero_fraction * ((unique_depth - i)
                         / static_cast<bst_float>(unique_depth + 1));
    } else if (zero_fraction != 0) {
      total += (unique_path[i].pweight / zero_fraction) / ((unique_depth - i)
               / static_cast<bst_float>(unique_depth + 1));
    } else {
      CHECK_EQ(unique_path[i].pweight, 0)
        << "Unique path " << i << " must have zero weight";
    }
  }
  return total;
}

// recursive computation of SHAP values for a decision tree
void RegTree::TreeShap(const RegTree::FVec &feat, bst_float *phi,
                       unsigned node_index, unsigned unique_depth,
                       PathElement *parent_unique_path,
                       bst_float parent_zero_fraction,
                       bst_float parent_one_fraction, int parent_feature_index,
                       int condition, unsigned condition_feature,
                       bst_float condition_fraction) const {
  const auto node = (*this)[node_index];

  // stop if we have no weight coming down to us
  if (condition_fraction == 0) return;

  // extend the unique path
  PathElement *unique_path = parent_unique_path + unique_depth + 1;
  std::copy(parent_unique_path, parent_unique_path + unique_depth + 1, unique_path);

  if (condition == 0 || condition_feature != static_cast<unsigned>(parent_feature_index)) {
    ExtendPath(unique_path, unique_depth, parent_zero_fraction,
               parent_one_fraction, parent_feature_index);
  }
  const unsigned split_index = node.SplitIndex();

  // leaf node
  if (node.IsLeaf()) {
    for (unsigned i = 1; i <= unique_depth; ++i) {
      const bst_float w = UnwoundPathSum(unique_path, unique_depth, i);
      const PathElement &el = unique_path[i];
      phi[el.feature_index] += w * (el.one_fraction - el.zero_fraction)
                                 * node.LeafValue() * condition_fraction;
    }

  // internal node
  } else {
    // find which branch is "hot" (meaning x would follow it)
    unsigned hot_index = 0;
    if (feat.IsMissing(split_index)) {
      hot_index = node.DefaultChild();
    } else if (feat.Fvalue(split_index) < node.SplitCond()) {
      hot_index = node.LeftChild();
    } else {
      hot_index = node.RightChild();
    }
    const unsigned cold_index = (static_cast<int>(hot_index) == node.LeftChild() ?
                                 node.RightChild() : node.LeftChild());
    const bst_float w = this->Stat(node_index).sum_hess;
    const bst_float hot_zero_fraction = this->Stat(hot_index).sum_hess / w;
    const bst_float cold_zero_fraction = this->Stat(cold_index).sum_hess / w;
    bst_float incoming_zero_fraction = 1;
    bst_float incoming_one_fraction = 1;

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    unsigned path_index = 0;
    for (; path_index <= unique_depth; ++path_index) {
      if (static_cast<unsigned>(unique_path[path_index].feature_index) == split_index) break;
    }
    if (path_index != unique_depth + 1) {
      incoming_zero_fraction = unique_path[path_index].zero_fraction;
      incoming_one_fraction = unique_path[path_index].one_fraction;
      UnwindPath(unique_path, unique_depth, path_index);
      unique_depth -= 1;
    }

    // divide up the condition_fraction among the recursive calls
    bst_float hot_condition_fraction = condition_fraction;
    bst_float cold_condition_fraction = condition_fraction;
    if (condition > 0 && split_index == condition_feature) {
      cold_condition_fraction = 0;
      unique_depth -= 1;
    } else if (condition < 0 && split_index == condition_feature) {
      hot_condition_fraction *= hot_zero_fraction;
      cold_condition_fraction *= cold_zero_fraction;
      unique_depth -= 1;
    }

    TreeShap(feat, phi, hot_index, unique_depth + 1, unique_path,
             hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
             split_index, condition, condition_feature, hot_condition_fraction);

    TreeShap(feat, phi, cold_index, unique_depth + 1, unique_path,
             cold_zero_fraction * incoming_zero_fraction, 0,
             split_index, condition, condition_feature, cold_condition_fraction);
  }
}

void RegTree::CalculateContributions(const RegTree::FVec &feat,
                                     unsigned root_id, bst_float *out_contribs,
                                     int condition,
                                     unsigned condition_feature) const {
  // find the expected value of the tree's predictions
  if (condition == 0) {
    bst_float node_value = this->node_mean_values_[static_cast<int>(root_id)];
    out_contribs[feat.Size()] += node_value;
  }

  // Preallocate space for the unique path data
  const int maxd = this->MaxDepth(root_id) + 2;
  auto *unique_path_data = new PathElement[(maxd * (maxd + 1)) / 2];

  TreeShap(feat, out_contribs, root_id, 0, unique_path_data,
           1, 1, -1, condition, condition_feature, 1);
  delete[] unique_path_data;
}
}  // namespace xgboost
