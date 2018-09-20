/*!
 * Copyright 2018 by Contributors
 * \file split_evaluator.cc
 * \brief Contains implementations of different split evaluators.
 */
#include "split_evaluator.h"
#include <dmlc/json.h>
#include <dmlc/registry.h>
#include <algorithm>
#include <unordered_set>
#include <vector>
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
  // maximum delta update we can add in weight estimation
  // this parameter can be used to stabilize update
  // default=0 means no constraint on weight delta
  float max_delta_step;

  DMLC_DECLARE_PARAMETER(ElasticNetParams) {
    DMLC_DECLARE_FIELD(reg_lambda)
      .set_lower_bound(0.0)
      .set_default(1.0)
      .describe("L2 regularization on leaf weight");
    DMLC_DECLARE_FIELD(reg_alpha)
      .set_lower_bound(0.0)
      .set_default(0.0)
      .describe("L1 regularization on leaf weight");
    DMLC_DECLARE_FIELD(max_delta_step)
      .set_lower_bound(0.0f)
      .set_default(0.0f)
      .describe("Maximum delta step we allow each tree's weight estimate to be. "\
                "If the value is set to 0, it means there is no constraint");
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
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
        + 2.0 * params_.reg_alpha * std::abs(weight);
    return -loss;
  }

  bst_float ComputeScore(bst_uint parentID, const GradStats &stats) const {
    if (params_.max_delta_step == 0.0f) {
      return Sqr(ThresholdL1(stats.sum_grad)) / (stats.sum_hess + params_.reg_lambda);
    } else {
      return ComputeScore(parentID, stats, ComputeWeight(parentID, stats));
    }
  }

  bst_float ComputeWeight(bst_uint parentID, const GradStats& stats)
      const override {
    bst_float w = -ThresholdL1(stats.sum_grad) / (stats.sum_hess + params_.reg_lambda);
    if (params_.max_delta_step != 0.0f && std::abs(w) > params_.max_delta_step) {
      w = std::copysign(params_.max_delta_step, w);
    }
    return w;
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
.describe("Use an elastic net regulariser")
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
  std::string interaction_constraints;
  bst_uint num_feature;

  DMLC_DECLARE_PARAMETER(InteractionConstraintParams) {
    DMLC_DECLARE_FIELD(interaction_constraints)
        .set_default("")
        .describe("Constraints for interaction representing permitted interactions."
                  "The constraints must be specified in the form of a nest list,"
                  "e.g. [[0, 1], [2, 3, 4]], where each inner list is a group of"
                  "indices of features that are allowed to interact with each other."
                  "See tutorial for more information");
    DMLC_DECLARE_FIELD(num_feature)
        .describe("Number of total features used");
  }
};

DMLC_REGISTER_PARAMETER(InteractionConstraintParams);

/*! \brief Enforces that the tree is monotonically increasing/decreasing with respect to a user specified set of
      features.
*/
class InteractionConstraint final : public SplitEvaluator {
 public:
  explicit InteractionConstraint(std::unique_ptr<SplitEvaluator> inner) {
    if (!inner) {
      LOG(FATAL) << "InteractionConstraint must be given an inner evaluator";
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
    if (params_.interaction_constraints.empty()) {
      return;  // short-circuit if no constraint is specified
    }

    // Parse interaction constraints
    std::istringstream iss(params_.interaction_constraints);
    dmlc::JSONReader reader(&iss);
    // Read std::vector<std::vector<bst_uint>> first and then
    //   convert to std::vector<std::unordered_set<bst_uint>>
    std::vector<std::vector<bst_uint>> tmp;
    reader.Read(&tmp);
    for (const auto& e : tmp) {
      interaction_constraints_.emplace_back(e.begin(), e.end());
    }

    // Initialise interaction constraints record with all variables permitted for the first node
    int_cont_.clear();
    int_cont_.resize(1, std::unordered_set<bst_uint>());
    int_cont_[0].reserve(params_.num_feature);
    for (bst_uint i = 0; i < params_.num_feature; ++i) {
      int_cont_[0].insert(i);
    }

    // Initialise splits record
    splits_.clear();
    splits_.resize(1, std::unordered_set<bst_uint>());
  }

  SplitEvaluator* GetHostClone() const override {
    if (params_.interaction_constraints.empty()) {
      // No interaction constraints specified, just return a clone of inner
      return inner_->GetHostClone();
    } else {
      auto c = new InteractionConstraint(
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
    // Return negative infinity score if feature is not permitted by interaction constraints
    if (!CheckInteractionConstraint(featureid, nodeid)) {
      return -std::numeric_limits<bst_float>::infinity();
    }

    // Otherwise, get score from inner evaluator
    bst_float score = inner_->ComputeSplitScore(
      nodeid, featureid, left_stats, right_stats, left_weight, right_weight);
    return score;
  }

  bst_float ComputeScore(bst_uint parentID, const GradStats& stats, bst_float weight)
      const override {
    return inner_->ComputeScore(parentID, stats, weight);
  }

  bst_float ComputeWeight(bst_uint parentID, const GradStats& stats)
      const override {
    return inner_->ComputeWeight(parentID, stats);
  }

  void AddSplit(bst_uint nodeid,
                bst_uint leftid,
                bst_uint rightid,
                bst_uint featureid,
                bst_float leftweight,
                bst_float rightweight) override {
    inner_->AddSplit(nodeid, leftid, rightid, featureid, leftweight, rightweight);

    if (params_.interaction_constraints.empty()) {
      return;  // short-circuit if no constraint is specified
    }
    bst_uint newsize = std::max(leftid, rightid) + 1;

    // Record previous splits for child nodes
    std::unordered_set<bst_uint> feature_splits = splits_[nodeid];  // fid history of current node
    feature_splits.insert(featureid);  // add feature of current node
    splits_.resize(newsize);
    splits_[leftid] = feature_splits;
    splits_[rightid] = feature_splits;

    // Resize constraints record, initialise all features to be not permitted for new nodes
    int_cont_.resize(newsize, std::unordered_set<bst_uint>());

    // Permit features used in previous splits
    for (bst_uint fid : feature_splits) {
      int_cont_[leftid].insert(fid);
      int_cont_[rightid].insert(fid);
    }

    // Loop across specified interactions in constraints
    for (const auto& constraint : interaction_constraints_) {
      bst_uint flag = 1;  // flags whether the specified interaction is still relevant

      // Test relevance of specified interaction by checking all previous features are included
      for (bst_uint checkvar : feature_splits) {
        if (constraint.count(checkvar) == 0) {
          flag = 0;
          break;  // interaction is not relevant due to unmet constraint
        }
      }

      // If interaction is still relevant, permit all other features in the interaction
      if (flag == 1) {
        for (bst_uint k : constraint) {
          int_cont_[leftid].insert(k);
          int_cont_[rightid].insert(k);
        }
      }
    }
  }

 private:
  InteractionConstraintParams params_;
  std::unique_ptr<SplitEvaluator> inner_;
  // interaction_constraints_[constraint_id] contains a single interaction
  //   constraint, which specifies a group of feature IDs that can interact
  //   with each other
  std::vector< std::unordered_set<bst_uint> > interaction_constraints_;
  // int_cont_[nid] contains the set of all feature IDs that are allowed to
  //   be used for a split at node nid
  std::vector< std::unordered_set<bst_uint> > int_cont_;
  // splits_[nid] contains the set of all feature IDs that have been used for
  //   splits in node nid and its parents
  std::vector< std::unordered_set<bst_uint> > splits_;

  // Check interaction constraints. Returns true if a given feature ID is
  //   permissible in a given node; returns false otherwise
  inline bool CheckInteractionConstraint(bst_uint featureid, bst_uint nodeid) const {
    // short-circuit if no constraint is specified
    return (params_.interaction_constraints.empty()
            || int_cont_[nodeid].count(featureid) > 0);
  }
};

XGBOOST_REGISTER_SPLIT_EVALUATOR(InteractionConstraint, "interaction")
.describe("Enforces interaction constraints on tree features")
.set_body([](std::unique_ptr<SplitEvaluator> inner) {
    return new InteractionConstraint(std::move(inner));
  });

}  // namespace tree
}  // namespace xgboost
