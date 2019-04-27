#ifndef GIBBS_UPDATER_H_
#define GIBBS_UPDATER_H_

#include <cmath>
#include <memory>
#include <vector>
#include <map>
#include <queue>

#include <dmlc/parameter.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/tree_model.h>
#include <xgboost/predictor.h>

#include "rv.h"
#include "pool.h"
#include "../../src/common/timer.h"

namespace xgboost {

struct GibbsParam : public dmlc::Parameter<GibbsParam> {
  // control splits
  bst_float alpha;
  bst_float beta;

  bst_float mu;
  bst_float sigma;

  bst_uint num_trees;

  bst_float nu;
  bst_float lambda;
  bst_uint burn_in;

  DMLC_DECLARE_PARAMETER(GibbsParam) {
    DMLC_DECLARE_FIELD(alpha)
        .set_range(0, 1)
        .set_default(0.95)
        .describe("Controls tree structure");
    DMLC_DECLARE_FIELD(beta)
        .set_default(2.0f)
        .describe("Controls tree structure");

    DMLC_DECLARE_FIELD(mu)
        .set_default(0.0f);
    DMLC_DECLARE_FIELD(sigma)
        .set_default(0.2);

    DMLC_DECLARE_FIELD(num_trees)
        .set_default(100)
        .describe("Number of trees.");

    DMLC_DECLARE_FIELD(nu)
        .set_default(3.0f);  // FIXME: Set prior
    DMLC_DECLARE_FIELD(lambda)
        .set_default(1.0f);  // FIXME: Set prior
    DMLC_DECLARE_FIELD(burn_in)
        .set_lower_bound(0)
        .set_default(100)
        .describe("Number of iterations for burn in.");
  }
};

class TreeMutation;

class GibbsUpdater {
  enum class MoveKind {
    kGrow,
    kPrune,
    kSwap,
    kChange
  };

 public:
  /* \brief One sample in the chain. */
  using SigleTreeChain = std::vector<std::unique_ptr<TreeMutation>>;
  /* \brief Sampling chain for all trees. */
  struct TreesChain : public std::vector<SigleTreeChain> {
    std::unique_ptr<TreeMutation>& lastTree(int32_t tree_idx) {
      return this->at(tree_idx).back();
    }
    void add(int32_t tree_idx, std::unique_ptr<TreeMutation>&& p_tree) {
      this->at(tree_idx).emplace_back(std::move(p_tree));
    }
  };

  /*
   * \brief Implement the diffing trick from bart-machine
   *
   * See `bartMachine: Machine Learning with Bayesian Additive Regression Trees'
   * Section 3.1 and corresponding source code.
   */
  std::vector<bst_float> _running_sum;

  void update(DMatrix* dmat, std::vector<std::shared_ptr<RegTree>>& trees);

  GibbsUpdater();
  void configure(const std::vector<std::pair<std::string, std::string>>& args);

 protected:
  void init(DMatrix* p_fmat, std::vector<std::shared_ptr<RegTree>>& trees);
  void initData(DMatrix* p_fmat);
  void initTrees();

  /*
   * \brief Compute posterior for sigma.
   */
  void drawSigma(DMatrix* dmat, TreeMutation& tree);

  void computeResponse(size_t tree_id, std::vector<float> const& labels);
  void updateTrees(DMatrix* p_fmat);

 protected:
  Uniform _move_chooser;

  InverseGamma _sigma_sampler;
  float _sigma;
  std::vector<InverseGamma> _sigma_chain;

  bst_float _y_min;
  bst_float _y_max;

  GibbsParam _param;
  common::Monitor _monitor;

  TreesChain _chain;
  std::vector<bst_float> _responses_cache;
  DMatrix* _p_fmat;

  bool _initialized;
};

class TreeMutation : public RegTree {
 public:
  static constexpr int32_t kRejected {-1};

 protected:
  std::vector<std::vector<size_t>> _indices;
  // FIXME(trivialfis): Move out to updater
  std::vector<float> _responses;
  // FIXME(trivialfis): Move out to updater
  std::vector<float> _predictions;

  PersistentPool<int32_t> _splitable_nodes;
  PersistentPool<int32_t> _splitable_features;

  PersistentPool<int32_t> _prunable_nodes;

  int32_t _tree_id;
  GibbsUpdater::TreesChain* _chain;

 protected:
  bst_float growTransitionRatio(int32_t nid);
  bst_float growLikelihoodRatio(float sigma, float sigma_mu,
                                int32_t parent, int32_t left, int32_t right);
  bst_float growTreeStructureRatio(int32_t parent,
                                   float const alpha, float const beta);

 public:
  TreeMutation(GibbsUpdater::TreesChain* chain, int32_t tree_id,
               DMatrix const* p_fmat) :
      _chain{chain}, _tree_id{tree_id}
  {
    _splitable_nodes.push(0);  // root

    auto const& info = p_fmat->Info();
    _splitable_features = PersistentPool<int32_t>(info.num_col_);

    _predictions.resize(info.num_row_, 0);
    _responses.resize(info.num_row_, 0);

    _indices.resize(1);
    _indices[0].resize(info.num_row_);
    for (size_t i = 0; i < info.num_row_; ++i) {
      _indices[0][i] = i;
    }
  }

  TreeMutation& operator=(TreeMutation const& other) {
    // from regtree
    this->nodes_ = other.nodes_;
    this->deleted_nodes_ = other.deleted_nodes_;
    this->node_mean_values_ = other.node_mean_values_;
    this->stats_ = other.stats_;

    // extended data field
    this->_indices = other._indices;
    this->_responses = other._responses;
    this->_predictions = other._responses;
    this->_splitable_nodes = other._splitable_nodes;
    this->_splitable_features = other._splitable_features;
    this->_prunable_nodes = other._prunable_nodes;

    this->_chain = other._chain;
    return *this;
  }

  TreeMutation& operator=(TreeMutation&& other) {
    // from regtree
    this->nodes_ = std::move(other.nodes_);
    this->deleted_nodes_ = std::move(other.deleted_nodes_);
    this->node_mean_values_ = std::move(other.node_mean_values_);
    this->stats_ = std::move(other.stats_);

    // extended data field
    this->_indices = std::move(other._indices);
    this->_responses = std::move(other._responses);
    this->_predictions = std::move(other._responses);
    this->_splitable_nodes = std::move(other._splitable_nodes);
    this->_splitable_features = std::move(other._splitable_features);
    this->_prunable_nodes = std::move(other._prunable_nodes);

    this->_chain = other._chain;
    return *this;
  }

  bst_float growAcceptanceRatio(
      int32_t parent, int32_t left, int32_t right,
      float sigma,
      GibbsParam const& param) {

    auto transition = growTransitionRatio(parent);
    auto likelihood = growLikelihoodRatio(sigma, param.sigma, parent, left, right);
    auto structure = growTreeStructureRatio(parent, param.alpha, param.beta);

    float ratio = std::min(0.0f,
                           transition + likelihood + structure);
    return ratio;
  }

  int32_t grow(float sigma, GibbsParam param, DMatrix* p_fmat);
  int32_t prune(float sigma, GibbsParam param, DMatrix* p_fmat);

  bst_float sumResponsesByNode(int32_t nid) {
    auto& node_indiecs = _indices.at(nid);
    float sum = 0;
    for (size_t i = 0; i < node_indiecs.size(); ++i) {
      float value = _responses.at(node_indiecs.at(i));
      sum += value;
    }
    return sum;
  }
  /*
   * \brief sample from leafs' posterior distribution
   */
  Normal leafValueDistribution(bst_float sigma,
                               int32_t nid,
                               GibbsParam const& param) {
    float sum = sumResponsesByNode(nid);
    auto const loc = (sum / (_indices.at(nid).size() * param.sigma + sigma));
    auto const scale = (sigma * param.sigma /
                        (_indices.at(nid).size() * param.sigma + sigma));
    auto const leaf = Normal(loc, scale);
    return leaf;
  }

  void applySplit(int32_t node_id, int32_t split_index, float split_cond,
                  DMatrix* p_fmat) {
    CHECK_GE(node_id, 0);
    _splitable_features.erase(split_index);
    _splitable_nodes.erase(node_id);

    // assign split
    int const pleft = this->AllocNode();
    int const pright = this->AllocNode();
    CHECK(!(*this)[pleft].IsDeleted()) << pleft;
    CHECK(!(*this)[pright].IsDeleted()) << pright;
    (*this)[pleft].SetParent(node_id);
    (*this)[node_id].SetLeftChild(pleft);
    (*this)[pright].SetParent(node_id);
    (*this)[node_id].SetRightChild(pright);

    (*this)[pleft].SetLeaf(std::numeric_limits<bst_float>::infinity());
    (*this)[pright].SetLeaf(std::numeric_limits<bst_float>::infinity());

    auto& node = (*this)[node_id];

    _prunable_nodes.push(node_id);
    if (!node.IsRoot() && _indices.at(node.Parent()).size() >= 2) {
      // the original parent is prunable.
      _prunable_nodes.erase(node.Parent());
    }

    std::vector<size_t> left_indices;
    std::vector<size_t> right_indices;
    auto& info = p_fmat->Info();
    for (auto batch : p_fmat->GetRowBatches()) {
      for (size_t row_id = 0; row_id < info.num_row_; ++row_id) {
        auto row_inst = batch[row_id];
        for (auto& entry : row_inst) {
          // with unknow column indices store in an Entry instead of
          // standard CSR format, we have to walk through all values
          // to get the wanted entry
          if (entry.index == split_index) {
            if (entry.fvalue < split_cond) {
              left_indices.emplace_back(row_id);
            } else {
              right_indices.emplace_back(row_id);
            }
          }
        }
      }
    }

    int32_t left_id = (*this)[node_id].LeftChild();
    CHECK_GE(left_id, 0);
    int32_t right_id = (*this)[node_id].RightChild();
    CHECK_GE(right_id, 0);
    size_t max_id = std::max(std::max(left_id, right_id), node_id);
    if (_indices.size() < max_id) {
      _indices.resize(max_id + 1);
    }
    _indices.at(left_id) = std::move(left_indices);
    _indices.at(right_id) = std::move(right_indices);
  }

  void applyPrune(int32_t node_id, int32_t split_index) {
    _splitable_features.push(split_index);
    _splitable_nodes.push(node_id);

    _prunable_nodes.erase(node_id);
    auto & node = (*this)[node_id];
    if (!node.IsRoot() && _indices.at(node.Parent()).size() >= 2) {
      _prunable_nodes.push(node.Parent());
    }

    _indices.erase(_indices.begin() + node.LeftChild());
    _indices.erase(_indices.begin() + node.RightChild());

    // assign value in leaf sampling
    CHECK_GT(nodes_.size(), node_id);
    this->CollapseToLeaf(node_id, std::numeric_limits<float>::infinity());
  }

  void sampleLeaf(int32_t last_node_id,
                  float sigma,
                  GibbsParam const& param) {
    auto& node = (*this)[last_node_id];

    if (node.IsLeaf()) {
      auto value = leafValueDistribution(sigma, last_node_id,  param).sample();
      (*this)[last_node_id].SetLeaf(value);
      for (auto ind : _indices.at(last_node_id)) {
        _predictions.at(ind) = value;
      }
    } else {
      int32_t left_id = node.LeftChild();
      int32_t right_id = node.RightChild();

      auto left_value = leafValueDistribution(sigma, left_id,  param).sample();
      auto right_value = leafValueDistribution(sigma, right_id, param).sample();

      (*this)[left_id].SetLeaf(left_value);
      (*this)[right_id].SetLeaf(right_value);

      auto& left_node = RegTree::nodes_.at(left_id);
      auto& right_node = RegTree::nodes_.at(right_id);

      for (auto ind : _indices[left_id]) {
        _predictions.at(ind) = left_node.LeafValue();
      }
      for (auto ind : _indices[right_id]) {
        _predictions.at(ind) = right_node.LeafValue();
      }
    }
  }

  void updateResponse(std::vector<bst_float> const& new_responses) {
    for (size_t i = 0; i < _responses.size(); ++i) {
      _responses.at(i) = new_responses.at(i);
    }
  }

  decltype(_responses)& getResponses() {
    return _responses;
  }
  decltype(_predictions)& getPredictionCache() {
    return _predictions;
  }
};

}  // xgboost

#endif  // GIBBS_UPDATER_H_
