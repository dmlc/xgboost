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
                 const RegressionTree& tree,
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

std::string RegressionTree::DumpModel(const FeatureMap& fmap,
                               bool with_stats,
                               std::string format) const {
  std::stringstream fo("");
  DumpRegTree(fo, *this, fmap, 0, 0, false, with_stats, format);
  return fo.str();
}

/*! \brief get next position of the tree given current pid */
int RegressionTree::GetNext(int pid, bst_float fvalue, bool is_unknown) const {
  bst_float split_value = (*this).GetNode(pid).SplitCond();
  if (is_unknown) {
    return (*this).GetNode(pid).DefaultChild();
  } else {
    if (fvalue < split_value) {
      return (*this).GetNode(pid).LeftChild();
    } else {
      return (*this).GetNode(pid).RightChild();
    }
  }
}

// do not need to read if only use the model
void RegressionTree::FVec::Init(size_t size) {
  Entry e; e.flag = -1;
  data_.resize(size);
  std::fill(data_.begin(), data_.end(), e);
}

void RegressionTree::FVec::Fill(const SparsePage::Inst& inst) {
  for (bst_uint i = 0; i < inst.size(); ++i) {
    if (inst[i].index >= data_.size()) continue;
    data_[inst[i].index].fvalue = inst[i].fvalue;
  }
}

void RegressionTree::FVec::Drop(const SparsePage::Inst& inst) {
  for (bst_uint i = 0; i < inst.size(); ++i) {
    if (inst[i].index >= data_.size()) continue;
    data_[inst[i].index].flag = -1;
  }
}

size_t RegressionTree::FVec::Size() const {
  return data_.size();
}

bst_float RegressionTree::FVec::Fvalue(size_t i) const {
  return data_[i].fvalue;
}

bool RegressionTree::FVec::IsMissing(size_t i) const {
  return data_[i].flag == -1;
}

int RegressionTree::GetLeafIndex(const RegressionTree::FVec& feat, unsigned root_id) const {
  auto pid = static_cast<int>(root_id);
  while (!(*this).GetNode(pid).IsLeaf()) {
    unsigned split_index = (*this).GetNode(pid).SplitIndex();
    pid = this->GetNext(pid, feat.Fvalue(split_index), feat.IsMissing(split_index));
  }
  return pid;
}

float RegressionTree::GetNodeMeanValue(int nid) {
  size_t num_nodes = this->param.num_nodes;
  if (this->node_mean_values_.size() != num_nodes) {
    this->node_mean_values_.resize(num_nodes);
    this->FillNodeMeanValue(0);
  }
  return this->node_mean_values_[nid];
}

bst_float RegressionTree::FillNodeMeanValue(int nid) {
  bst_float result;
  const auto& node = (*this).GetNode(nid);
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

void RegressionTree::CalculateContributionsApprox(const RegressionTree::FVec& feat, unsigned root_id,
                                                  bst_float *out_contribs) {
  // this follows the idea of http://blog.datadive.net/interpreting-random-forests/
  unsigned split_index = 0;
  auto pid = static_cast<int>(root_id);
  // update bias value
  bst_float node_value = this->GetNodeMeanValue(pid);
  out_contribs[feat.Size()] += node_value;
  if ((*this).GetNode(pid).IsLeaf()) {
    // nothing to do anymore
    return;
  }
  while (!(*this).GetNode(pid).IsLeaf()) {
    split_index = (*this).GetNode(pid).SplitIndex();
    pid = this->GetNext(pid, feat.Fvalue(split_index), feat.IsMissing(split_index));
    bst_float new_value = this->GetNodeMeanValue(pid);
    // update feature weight
    out_contribs[split_index] += new_value - node_value;
    node_value = new_value;
  }
  bst_float leaf_value = (*this).GetNode(pid).LeafValue();
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
                       bst_float zero_fraction, bst_float one_fraction, int feature_index) {
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
void UnwindPath(PathElement *unique_path, unsigned unique_depth, unsigned path_index) {
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
    } else {
      total += (unique_path[i].pweight / zero_fraction) / ((unique_depth - i)
               / static_cast<bst_float>(unique_depth + 1));
    }
  }
  return total;
}

// recursive computation of SHAP values for a decision tree
void RegressionTree::TreeShap(const RegressionTree::FVec& feat, bst_float *phi,
                              unsigned node_index, unsigned unique_depth,
                              PathElement *parent_unique_path, bst_float parent_zero_fraction,
                              bst_float parent_one_fraction, int parent_feature_index,
                              int condition, unsigned condition_feature,
                              bst_float condition_fraction) const {
  const auto node = this->GetNode(node_index);

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

void RegressionTree::CalculateContributions(const RegressionTree::FVec& feat, unsigned root_id,
                                            bst_float *out_contribs,
                                            int condition,
                                            unsigned condition_feature) {
  // find the expected value of the tree's predictions
  if (condition == 0) {
    bst_float node_value = this->GetNodeMeanValue(root_id);
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
