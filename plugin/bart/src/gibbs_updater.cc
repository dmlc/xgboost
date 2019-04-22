#include <dmlc/parameter.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/logging.h>
#include <xgboost/tree_model.h>

#include <cinttypes>

#include "gibbs_updater.h"
#include "rv.h"
#include "../../src/common/timer.h"

namespace xgboost {

DMLC_REGISTER_PARAMETER(GibbsParam);

bst_float TreeMutation::growTransitionRatio(int32_t nid) {
  CHECK_NE(this, _chain->at(_tree_id).back().get())
      << "Tree being sampled should not be put into the chain.";
  auto const& last_tree = _chain->lastTree(_tree_id);
  float const n_splitable_nodes = last_tree->_splitable_nodes.size();
  float const n_splitable_features = _splitable_features.size();
  // find number of internal nodes which have only two childern
  // terminal nodes.  Since every split has exact two children, that
  // would be all parents of leaf nodes.
  float const n_prunable_nodes = _prunable_nodes.size();

  // FIXME(trivialfis): For now we just assume the values are unique
  float const n_rows_in_node = _indices.at(nid).size() + kRtEps;
  float result = std::log(n_rows_in_node) + std::log(n_splitable_features)
                 + std::log(n_splitable_nodes) - std::log(n_prunable_nodes);
  return result;
}

bst_float TreeMutation::growLikelihoodRatio(float sigma, float sigma_mu,
                                            int32_t parent, int32_t left, int32_t right) {
  float const n_left_rows = _indices.at(left).size();
  float const n_right_rows = _indices.at(right).size();

  float parent_to_children = 0;
  {
    float parent = sigma * (sigma + n_left_rows * sigma_mu);
    float left = sigma + n_left_rows * sigma_mu;
    float right = sigma + n_right_rows * sigma_mu;

    parent_to_children = parent / (left * right);
    parent_to_children = std::sqrt(parent_to_children);
    parent_to_children = std::log(parent_to_children);
  }

  float exp_term = 0;
  {
    float left_response = 0;
    for (size_t i = 0; i < n_left_rows; ++i) {
      left_response += _responses[_indices[left][i]];
    }
    left_response *= left_response;
    float denominator = sigma + n_left_rows * sigma_mu + kRtEps;
    exp_term += left_response / denominator;
  }
  {
    float right_response = 0;
    for (size_t i  = 0; i < n_right_rows; ++i) {
      right_response += _responses[_indices[right][i]];
    }
    right_response *= right_response;
    float denominator = sigma + n_right_rows * sigma_mu + kRtEps;
    exp_term += right_response / denominator;
  }
  {
    float parent_response = 0.0f;
    for (size_t i = 0; i < _indices.at(parent).size(); ++i) {
      parent_response += _responses[_indices[parent][i]];
    }
    parent_response *= parent_response;
    float denominator = sigma + _indices.at(parent).size() * sigma_mu + kRtEps;
    exp_term -= parent_response / denominator;
  }
  float const multiplier = sigma_mu / (2 * sigma + kRtEps);
  exp_term *= multiplier;

  float result = parent_to_children + exp_term;
  return result;
}

bst_float TreeMutation::growTreeStructureRatio(int32_t parent,
                                               bst_float const alpha, bst_float const beta) {
  bst_float const depth = RegTree::GetDepth(parent);
  bst_float const n_splitable_nodes = _splitable_nodes.size() + kRtEps;
  bst_float const n_rows_in_node = _indices.at(parent).size() + kRtEps;

  bst_float result = std::log(alpha);
  result += 2.0f * std::log(1 - (alpha / std::pow(2+depth, beta)));
  result -= std::log(std::pow(1+depth, beta) - alpha);
  result -= std::log(n_splitable_nodes);
  result -= std::log(n_rows_in_node);

  return result;
}

int32_t TreeMutation::grow(float sigma, GibbsParam param, DMatrix* p_fmat) {
  if (_splitable_nodes.size() == 0) {
    LOG(DEBUG) << "No splitable node.";
    return kRejected;
  }
  // sample a node for split
  size_t const splitnode_pos = Uniform(0, _splitable_nodes.size()).sample();
  int32_t const split_nid = _splitable_nodes[splitnode_pos];
  CHECK((*this)[split_nid].IsLeaf())     << "split.nid: " << split_nid;
  CHECK(!(*this)[split_nid].IsDeleted()) << "split.nid: " << split_nid;

  // sample a column for split
  auto feature_id = static_cast<int32_t>(
      Uniform(0, _splitable_features.size()).sample());
  auto split_feature = _splitable_features[feature_id];

  // sample a value for split
  bst_float split_cond = 0;
  for (auto& batch : p_fmat->GetColumnBatches()) {
    auto column = batch[split_feature];
    CHECK_NE(column.size(), 0);
    auto max = column[column.size() - 1].fvalue;
    auto min = column[0].fvalue;
    CHECK_GE(min, max);
    split_cond = Uniform(min, max).sample();
  }

  this->applySplit(split_nid, split_feature, split_cond, p_fmat);
  return split_nid;
}

int32_t TreeMutation::prune(float sigma, GibbsParam param, DMatrix *p_fmat) {
  if (_prunable_nodes.size() == 0) {
    LOG(DEBUG) << "No prunable node found.";
    return kRejected;
  }
  // sample a node to prune
  size_t prune_pos = static_cast<size_t>(Uniform(0, _prunable_nodes.size()).sample());
  int32_t parent_id = _prunable_nodes[prune_pos];

  CHECK(!(*this)[parent_id].IsLeaf()) << "prune.id: " << parent_id;

  int32_t left_id = (*this)[parent_id].LeftChild();
  auto left_node = (*this)[left_id];

  int32_t right_id = (*this)[parent_id].RightChild();
  auto right_node = (*this)[right_id];

  applyPrune(parent_id, (*this)[parent_id].SplitIndex());
  return parent_id;
}

void GibbsUpdater::updateTrees(DMatrix* p_fmat) {
  auto& labels = p_fmat->Info().labels_.HostVector();
  for (int32_t tree_id = 0; tree_id < _chain.size(); ++tree_id) {
    computeResponse(tree_id, labels);
    auto& tree = _chain.lastTree(tree_id);  // the last tree
    TreeMutation candidate{ &_chain, tree_id, p_fmat };
    candidate = *tree;
    candidate.updateResponse(_responses_cache);

    // choose action
    auto action_id = _move_chooser.sample();
    auto action = static_cast<MoveKind>(action_id);
    if (tree->NumExtraNodes() == 0) {
      // force grow
      action = MoveKind::kGrow;
    }
    int32_t changed_nid = TreeMutation::kRejected;
    switch(action) {
      case MoveKind::kGrow: {
        int32_t split_nid = candidate.grow(_sigma, _param, p_fmat);
        if (split_nid == TreeMutation::kRejected) {
          break;
        }
        int32_t left_id = candidate[split_nid].LeftChild();
        int32_t right_id = candidate[split_nid].RightChild();

        float const ratio =
            candidate.growAcceptanceRatio(split_nid, left_id, right_id, _sigma, _param);
        bool accept = std::log(Uniform(0.0f, 1.0f).sample()) < ratio;

        if (accept) {
          changed_nid = split_nid;
        } else {
          LOG(DEBUG) << "Grow proposal rejected.";
          changed_nid = TreeMutation::kRejected;
        }
      }
        break;
      case MoveKind::kPrune: {
        int32_t prune_nid = candidate.prune(_sigma, _param, p_fmat);
        if (prune_nid == TreeMutation::kRejected) {
          break;
        }
        int32_t left_id = candidate[prune_nid].LeftChild();
        int32_t right_id = candidate[prune_nid].RightChild();
        float const ratio = -candidate.growAcceptanceRatio(
            prune_nid, left_id, right_id, _sigma, _param);
        bool accept = std::log(Uniform(0.0f, 1.0f).sample()) < ratio;
        if (accept) {
          changed_nid = prune_nid;
        } else {
          LOG(DEBUG) << "prune proposal rejected.";
          changed_nid = TreeMutation::kRejected;
        }
      }
        break;
      case MoveKind::kChange:
        LOG(FATAL) << "Not implemented.";
        break;
      case MoveKind::kSwap:
        LOG(FATAL) << "Not implemented.";
        break;
    }

    if (changed_nid != TreeMutation::kRejected) {
      auto& old_predictions = tree->getPredictionCache();
      CHECK_EQ(old_predictions.size(), _running_sum.size());
      for (size_t i = 0; i < _running_sum.size(); ++i) {
        _running_sum[i] -= old_predictions[i];
      }

      candidate.sampleLeaf(changed_nid, _sigma, _param);
      _chain.at(tree_id).emplace_back(
          std::unique_ptr<TreeMutation>{new TreeMutation{&_chain, tree_id, p_fmat}});
      *_chain.lastTree(tree_id) = candidate;

      auto& predictions = candidate.getPredictionCache();
      for (size_t i = 0; i < _running_sum.size(); ++i) {
        _running_sum[i] += predictions[i];
      }
    } else {
      std::unique_ptr<TreeMutation> copied_tree { new TreeMutation{&_chain, tree_id, p_fmat} };
      // Copy the tree first, `emplace_back` might destroy the pointer during resize.
      *copied_tree = *tree;
      CHECK_GT(_chain.size(), tree_id);
      // push the old tree
      _chain.add(tree_id, std::move(copied_tree));
    }
  }
}

void GibbsUpdater::computeResponse(size_t tree_id, std::vector<float> const& labels) {
  _responses_cache.resize(labels.size());
  auto& trees = _chain.back();
  auto predictions = trees.at(tree_id)->getPredictionCache();
  for (size_t i = 0; i < _responses_cache.size(); ++i) {
    _responses_cache[i] = labels [i] - _running_sum[i] + predictions[i];
  }
}

std::pair<float, float> normaliseLabels(HostDeviceVector<bst_float>* y) {
  auto& h_y = y->HostVector();
  auto min_iter = std::min_element(h_y.cbegin(), h_y.cend());
  auto max_iter = std::max_element(h_y.cbegin(), h_y.cend());

  bst_float min = *min_iter;
  bst_float max = *max_iter;

  bst_float diff = *max_iter - *min_iter;
  std::transform(h_y.begin(), h_y.end(), h_y.begin(),
                 [&](bst_float value) {
                   return (*min_iter + *max_iter) / 2;
                 });
  return std::make_pair(min, max);
}

void GibbsUpdater::initData(DMatrix* p_fmat) {
  auto& info = p_fmat->Info();
  // FIXME: Change to out of place.
  std::tie(_y_min, _y_max) = normaliseLabels(&info.labels_);
  _running_sum.resize(info.num_row_);

  // compute prior for sigma
  float shape = (_param.nu + info.num_row_) * 0.5;
  float rate = _param.nu * _param.lambda * 0.5;
  _sigma_sampler = InverseGamma(shape, rate);
}

void GibbsUpdater::initTrees() {
  _chain.resize(1);
  _chain.at(0).resize(_param.num_trees);
  auto& trees = _chain.at(0);

  int32_t tree_id = 0;
  for (auto& tree : trees) {
    // since we normalise labels, mean is just 0
    tree.reset(new TreeMutation{&_chain, tree_id, _p_fmat});
    (*tree)[0].SetLeaf(0);
    CHECK((*tree)[0].IsLeaf());
    tree_id++;
  }
}

void GibbsUpdater::init(DMatrix *p_fmat, std::vector<std::shared_ptr<RegTree>> &trees) {
  _p_fmat = p_fmat;
  if (!this->_initialized) {
    this->initTrees();
    this->initData(p_fmat);
    this->_initialized = true;
  }
}

void GibbsUpdater::drawSigma(DMatrix* p_fmat, TreeMutation& tree) {
  auto& info = p_fmat->Info();
  auto& prediction_cache = tree.getPredictionCache();
  auto& responses = tree.getResponses();
  CHECK_EQ(prediction_cache.size(), responses.size());
  auto sum_residue = 1.0f;

  for (int32_t i = 0; i < prediction_cache.size(); ++i) {
    responses[i] -= prediction_cache[i];
    sum_residue += responses[i] * responses[i];
  }

  float shape = _param.nu * info.num_row_ * 0.5;
  float rate = (_param.nu * _param.lambda + sum_residue) * 0.5;
  _sigma_sampler = InverseGamma(shape, rate);
  _sigma = _sigma_sampler.sample();
  _sigma_chain.push_back(_sigma);
}

void GibbsUpdater::update(DMatrix* dmat, std::vector<std::shared_ptr<RegTree>>& trees) {
  init(dmat, trees);
  updateTrees(dmat);
}

// FIXME(trivialfis): Implement complete proposal
GibbsUpdater::GibbsUpdater() :
    _move_chooser{0, 2}, _initialized{false} {
  _monitor.Init("Gibbs");
}

void GibbsUpdater::configure(
    const std::vector<std::pair<std::string, std::string> > &args) {
  _param.InitAllowUnknown(args);
}

}  // namespace xgboost
