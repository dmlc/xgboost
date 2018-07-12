/*!
 * Copyright 2018 by Contributors
 * \file split_evaluator.cc
 * \brief Contains implementations of different split evaluators.
 */
#include "split_evaluator.h"
#include <dmlc/registry.h>
#include <algorithm>
#include <limits>
#include <string>
#include <sstream>
#include <utility>
#include "param.h"
#include "../common/common.h"
#include "../common/host_device_vector.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::tree::SplitEvaluatorReg);
}  // namespace dmlc

namespace xgboost {
namespace tree {

SplitEvaluator* SplitEvaluator::Create(const std::string& name) {
  std::stringstream ss(name);
  std::string item;
  SplitEvaluator* eval = nullptr;
  // Construct a chain of SplitEvaluators. This allows one to specify multiple constraints.
  while (std::getline(ss, item, ',')) {
    auto* e = ::dmlc::Registry< ::xgboost::tree::SplitEvaluatorReg>
        ::Get()->Find(item);
    if (e == nullptr) {
      LOG(FATAL) << "Unknown SplitEvaluator " << name;
    }
    eval = (e->body)(std::unique_ptr<SplitEvaluator>(eval));
  }
  return eval;
}

// Default implementations of some virtual methods that aren't always needed
void SplitEvaluator::Init(
    const std::vector<std::pair<std::string, std::string> >& args) {}
void SplitEvaluator::Reset() {}
void SplitEvaluator::AddSplit(bst_uint nodeid,
                              bst_uint leftid,
                              bst_uint rightid,
                              bst_uint featureid,
                              bst_float leftweight,
                              bst_float rightweight) {}
bst_float SplitEvaluator::ComputeSplitScore(bst_uint nodeid,
                                            bst_uint featureid,
                                            const GradStats& left_stats,
                                            const GradStats& right_stats) const {
  bst_float left_weight = ComputeWeight(nodeid, left_stats);
  bst_float right_weight = ComputeWeight(nodeid, right_stats);
  return ComputeSplitScore(nodeid, featureid, left_stats, right_stats, left_weight, right_weight);
}

//! \brief Encapsulates the parameters for ElasticNet
struct ElasticNetParams : public dmlc::Parameter<ElasticNetParams> {
  bst_float reg_lambda;
  bst_float reg_alpha;
  bst_float reg_gamma;

  DMLC_DECLARE_PARAMETER(ElasticNetParams) {
    DMLC_DECLARE_FIELD(reg_lambda)
      .set_lower_bound(0.0)
      .set_default(1.0)
      .describe("L2 regularization on leaf weight");
    DMLC_DECLARE_FIELD(reg_alpha)
      .set_lower_bound(0.0)
      .set_default(0.0)
      .describe("L1 regularization on leaf weight");
    DMLC_DECLARE_FIELD(reg_gamma)
      .set_lower_bound(0.0)
      .set_default(0.0)
      .describe("Cost incurred by adding a new leaf node to the tree");
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
    DMLC_DECLARE_ALIAS(reg_gamma, gamma);
  }
};

DMLC_REGISTER_PARAMETER(ElasticNetParams);

/*! \brief Applies an elastic net penalty and per-leaf penalty. */
class ElasticNet final : public SplitEvaluator {
 public:
  explicit ElasticNet(std::unique_ptr<SplitEvaluator> inner) {
    if (inner) {
      LOG(FATAL) << "ElasticNet does not accept an inner SplitEvaluator";
    }
  }
  void Init(
      const std::vector<std::pair<std::string, std::string> >& args) override {
    params_.InitAllowUnknown(args);
  }

  SplitEvaluator* GetHostClone() const override {
    auto r = new ElasticNet(nullptr);
    r->params_ = this->params_;

    return r;
  }

  bst_float ComputeSplitScore(bst_uint nodeid,
                             bst_uint featureid,
                             const GradStats& left_stats,
                             const GradStats& right_stats,
                             bst_float left_weight,
                             bst_float right_weight) const override {
    return ComputeScore(nodeid, left_stats, left_weight) +
      ComputeScore(nodeid, right_stats, right_weight);
  }

  bst_float ComputeSplitScore(bst_uint nodeid,
                             bst_uint featureid,
                             const GradStats& left_stats,
                             const GradStats& right_stats) const override {
    return ComputeScore(nodeid, left_stats) + ComputeScore(nodeid, right_stats);
  }

  bst_float ComputeScore(bst_uint parentID, const GradStats &stats, bst_float weight)
      const override {
    auto loss = weight * (2.0 * stats.sum_grad + stats.sum_hess * weight
        + params_.reg_lambda * weight)
        + params_.reg_alpha * std::abs(weight);
    return -loss;
  }

  bst_float ComputeScore(bst_uint parentID, const GradStats &stats) const {
    return Sqr(ThresholdL1(stats.sum_grad)) / (stats.sum_hess + params_.reg_lambda);
  }

  bst_float ComputeWeight(bst_uint parentID, const GradStats& stats)
      const override {
    return -ThresholdL1(stats.sum_grad) / (stats.sum_hess + params_.reg_lambda);
  }

 private:
  ElasticNetParams params_;

  inline double ThresholdL1(double g) const {
    if (g > params_.reg_alpha) {
      return g - params_.reg_alpha;
    } else if (g < -params_.reg_alpha) {
      return g + params_.reg_alpha;
    } else {
      return 0.0;
    }
  }
};

XGBOOST_REGISTER_SPLIT_EVALUATOR(ElasticNet, "elastic_net")
.describe("Use an elastic net regulariser and a cost per leaf node")
.set_body([](std::unique_ptr<SplitEvaluator> inner) {
    return new ElasticNet(std::move(inner));
  });

/*! \brief Encapsulates the parameters required by the MonotonicConstraint
        split evaluator
*/
struct MonotonicConstraintParams
    : public dmlc::Parameter<MonotonicConstraintParams> {
  std::vector<bst_int> monotone_constraints;

  DMLC_DECLARE_PARAMETER(MonotonicConstraintParams) {
    DMLC_DECLARE_FIELD(monotone_constraints)
      .set_default(std::vector<bst_int>())
      .describe("Constraint of variable monotonicity");
  }
};

DMLC_REGISTER_PARAMETER(MonotonicConstraintParams);

/*! \brief Enforces that the tree is monotonically increasing/decreasing with respect to a user specified set of
      features.
*/
class MonotonicConstraint final : public SplitEvaluator {
 public:
  explicit MonotonicConstraint(std::unique_ptr<SplitEvaluator> inner) {
    if (!inner) {
      LOG(FATAL) << "MonotonicConstraint must be given an inner evaluator";
    }
    inner_ = std::move(inner);
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& args)
      override {
    inner_->Init(args);
    params_.InitAllowUnknown(args);
    Reset();
  }

  void Reset() override {
    lower_.resize(1, -std::numeric_limits<bst_float>::max());
    upper_.resize(1, std::numeric_limits<bst_float>::max());
  }

  SplitEvaluator* GetHostClone() const override {
    if (params_.monotone_constraints.size() == 0) {
      // No monotone constraints specified, just return a clone of inner to speed things up
      return inner_->GetHostClone();
    } else {
      auto c = new MonotonicConstraint(
        std::unique_ptr<SplitEvaluator>(inner_->GetHostClone()));
      c->params_ = this->params_;
      c->Reset();
      return c;
    }
  }

  bst_float ComputeSplitScore(bst_uint nodeid,
                             bst_uint featureid,
                             const GradStats& left_stats,
                             const GradStats& right_stats,
                             bst_float left_weight,
                             bst_float right_weight) const override {
    bst_float infinity = std::numeric_limits<bst_float>::infinity();
    bst_int constraint = GetConstraint(featureid);
    bst_float score = inner_->ComputeSplitScore(
      nodeid, featureid, left_stats, right_stats, left_weight, right_weight);

    if (constraint == 0) {
      return score;
    } else if (constraint > 0) {
      return left_weight <= right_weight ? score : -infinity;
    } else {
      return left_weight >= right_weight ? score : -infinity;
    }
  }

  bst_float ComputeScore(bst_uint parentID, const GradStats& stats, bst_float weight)
      const override {
    return inner_->ComputeScore(parentID, stats, weight);
  }

  bst_float ComputeWeight(bst_uint parentID, const GradStats& stats)
      const override {
    bst_float weight = inner_->ComputeWeight(parentID, stats);

    if (parentID == ROOT_PARENT_ID) {
      // This is the root node
      return weight;
    } else if (weight < lower_.at(parentID)) {
      return lower_.at(parentID);
    } else if (weight > upper_.at(parentID)) {
      return upper_.at(parentID);
    } else {
      return weight;
    }
  }

  void AddSplit(bst_uint nodeid,
                bst_uint leftid,
                bst_uint rightid,
                bst_uint featureid,
                bst_float leftweight,
                bst_float rightweight) override {
    inner_->AddSplit(nodeid, leftid, rightid, featureid, leftweight, rightweight);
    bst_uint newsize = std::max(leftid, rightid) + 1;
    lower_.resize(newsize);
    upper_.resize(newsize);
    bst_int constraint = GetConstraint(featureid);

    bst_float mid = (leftweight + rightweight) / 2;
    CHECK(!std::isnan(mid));
    CHECK(nodeid < upper_.size());

    upper_[leftid] = upper_.at(nodeid);
    upper_[rightid] = upper_.at(nodeid);
    lower_[leftid] = lower_.at(nodeid);
    lower_[rightid] = lower_.at(nodeid);

    if (constraint < 0) {
      lower_[leftid] = mid;
      upper_[rightid] = mid;
    } else if (constraint > 0) {
      upper_[leftid] = mid;
      lower_[rightid] = mid;
    }
  }

 private:
  MonotonicConstraintParams params_;
  std::unique_ptr<SplitEvaluator> inner_;
  std::vector<bst_float> lower_;
  std::vector<bst_float> upper_;

  inline bst_int GetConstraint(bst_uint featureid) const {
    if (featureid < params_.monotone_constraints.size()) {
      return params_.monotone_constraints[featureid];
    } else {
      return 0;
    }
  }
};

XGBOOST_REGISTER_SPLIT_EVALUATOR(MonotonicConstraint, "monotonic")
.describe("Enforces that the tree is monotonically increasing/decreasing "
    "w.r.t. specified features")
.set_body([](std::unique_ptr<SplitEvaluator> inner) {
    return new MonotonicConstraint(std::move(inner));
  });

/*! \brief Encapsulates the parameters required by the InteractionConstraint
        split evaluator
*/
struct InteractionConstraintParams
    : public dmlc::Parameter<InteractionConstraintParams> {
  std::vector<bst_int> int_constraints;
  bst_int nint_constraints;
  float reg_lambda;
  float reg_gamma;

  DMLC_DECLARE_PARAMETER(InteractionConstraintParams) {
    DMLC_DECLARE_FIELD(reg_lambda)
      .set_lower_bound(0.0)
      .set_default(1.0)
      .describe("L2 regularization on leaf weight");
    DMLC_DECLARE_FIELD(reg_gamma)
      .set_lower_bound(0.0f)
      .set_default(0.0f)
      .describe("Cost incurred by adding a new leaf node to the tree");
    DMLC_DECLARE_FIELD(int_constraints)
      .set_default(std::vector<bst_int>())
      .describe("Constraints for interactions representing permitted interactions");
    DMLC_DECLARE_FIELD(nint_constraints)
      .set_default(0)
      .describe("Number of specified interactions in int_constraints vector");
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_gamma, gamma);
  }
};

DMLC_REGISTER_PARAMETER(InteractionConstraintParams);

/*! \brief Enforces interaction constraints on tree features.*/
class InteractionConstraint final : public SplitEvaluator {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args)
      override {
    params_.InitAllowUnknown(args);
    Reset();
  }

  void Reset() override {
    // Number of features on the data, implied by the relative size of int_constraints vector and nint_constraints
    bst_uint nfeatures = params_.int_constraints.size() / params_.nint_constraints;

    // Initialise interaction constraints record with all variables permitted for the first node
    int_cont_.clear();
    int_cont_.resize(nfeatures, std::vector<bst_uint>(1, 1));

    // Initialise splits record
    splits_.clear();
    splits_.resize(1, std::vector<bst_uint>());
  }

  SplitEvaluator* GetHostClone() const override {
    if (params_.int_constraints.size() == 0) {
      // No interaction constraints specified, make a RidgePenalty evaluator
      using std::pair;
      using std::string;
      using std::to_string;
      using std::vector;
      auto c = new RidgePenalty();
      vector<pair<string, string> > args;
      args.emplace_back(
        pair<string, string>("reg_lambda", to_string(params_.reg_lambda)));
      args.emplace_back(
        pair<string, string>("reg_gamma", to_string(params_.reg_gamma)));
      c->Init(args);
      c->Reset();
      return c;
    } else {
      auto c = new InteractionConstraint();
      c->params_ = this->params_;
      c->Reset();
      return c;
    }
  }

  bst_float ComputeSplitScore(bst_uint nodeid,
                              bst_uint featureid,
                              const GradStats& left,
                              const GradStats& right) const override {
    // Return negative infinity score if feature is not permitted by interaction constraints
    bst_float infinity = std::numeric_limits<bst_float>::infinity();
    bst_uint constraint_int = GetIntConstraint(featureid, nodeid);
    if (constraint_int == 0) return -infinity;

    // parentID is not needed for this split evaluator. Just use 0.
    return ComputeScore(0, left) + ComputeScore(0, right);
  }

  bst_float ComputeScore(bst_uint parentID, const GradStats& stats)
      const override {
    return (stats.sum_grad * stats.sum_grad)
        / (stats.sum_hess + params_.reg_lambda) - params_.reg_gamma;
  }

  bst_float ComputeWeight(bst_uint parentID, const GradStats& stats)
      const override {
    return -stats.sum_grad / (stats.sum_hess + params_.reg_lambda);
  }

  void AddSplit(bst_uint nodeid,
                bst_uint leftid,
                bst_uint rightid,
                bst_uint featureid,
                bst_float leftweight,
                bst_float rightweight) override {
    bst_uint nfeatures = params_.int_constraints.size() / params_.nint_constraints;
    bst_uint newsize = std::max(leftid, rightid) + 1;

    // Record previous splits for child nodes
    std::vector<bst_uint> feature_splits = splits_[nodeid];  // previous features of current node
    feature_splits.resize(feature_splits.size() + 1, featureid);  // add feature of current node
    splits_.resize(newsize);
    splits_[leftid] = feature_splits;
    splits_[rightid] = feature_splits;

    // Resize constraints record, initialise all features to be not permitted for new nodes
    for (bst_uint i = 0; i < nfeatures; ++i) int_cont_[i].resize(newsize, 0);

    // Permit features used in previous splits
    for (bst_uint i = 0; i < feature_splits.size(); ++i){
      int_cont_[feature_splits[i]][leftid] = 1;
      int_cont_[feature_splits[i]][rightid] = 1;
    }

    // Loop across specified interactions in constraints
    for (bst_int i = 0; i < params_.nint_constraints; i++){
      bst_uint flag = 1;  // flags whether the specified interaction is still relevant

      // Test relevance of specified interaction by checking all previous features are included
      for (bst_uint j = 0; j < feature_splits.size(); j++){
        bst_uint checkvar = feature_splits[j];
        if (params_.int_constraints[nfeatures*i + checkvar] == 0) flag = 0;
      }

      // If interaction is still relevant, permit all other features in the interaction
      if (flag == 1) for (bst_uint k = 0; k < nfeatures; k ++){
        if (params_.int_constraints[nfeatures*i + k] != 0){
          int_cont_[k][leftid] = 1;
          int_cont_[k][rightid] = 1;
        }
      }
    }
  }

 private:
  InteractionConstraintParams params_;
  // Record of interaction constraints: (Per Feature) x (Per Node)
  std::vector< std::vector<bst_uint> > int_cont_;
  // Record of feature ids in previous splits: (Per Node) x (Number of Previous Splits)
  std::vector< std::vector<bst_uint> > splits_;

  inline bst_uint GetIntConstraint(bst_uint featureid, bst_uint nodeid) const {
    if (featureid < int_cont_.size()) {
      return int_cont_[featureid][nodeid];
    } else {
      return 1;
    }
  }
};

XGBOOST_REGISTER_SPLIT_EVALUATOR(InteractionConstraint, "interaction")
.describe("Enforces interaction constraints on tree features")
.set_body([]() {
    return new InteractionConstraint();
  });

}  // namespace tree
}  // namespace xgboost
