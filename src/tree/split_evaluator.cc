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
#include <utility>
#include "param.h"
#include "../common/common.h"
#include "../common/host_device_vector.h"

#define ROOT_PARENT_ID (-1 & ((1U << 31) - 1))

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::tree::SplitEvaluatorReg);
}  // namespace dmlc

namespace xgboost {
namespace tree {

SplitEvaluator* SplitEvaluator::Create(const std::string& name) {
  auto* e = ::dmlc::Registry< ::xgboost::tree::SplitEvaluatorReg>
      ::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown SplitEvaluator " << name;
  }
  return (e->body)();
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

//! \brief Encapsulates the parameters for by the RidgePenalty
struct RidgePenaltyParams : public dmlc::Parameter<RidgePenaltyParams> {
  float reg_lambda;
  float reg_gamma;

  DMLC_DECLARE_PARAMETER(RidgePenaltyParams) {
    DMLC_DECLARE_FIELD(reg_lambda)
      .set_lower_bound(0.0)
      .set_default(1.0)
      .describe("L2 regularization on leaf weight");
    DMLC_DECLARE_FIELD(reg_gamma)
      .set_lower_bound(0.0f)
      .set_default(0.0f)
      .describe("Cost incurred by adding a new leaf node to the tree");
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_gamma, gamma);
  }
};

DMLC_REGISTER_PARAMETER(RidgePenaltyParams);

/*! \brief Applies an L2 penalty and per-leaf penalty. */
class RidgePenalty final : public SplitEvaluator {
 public:
  void Init(
      const std::vector<std::pair<std::string, std::string> >& args) override {
    params_.InitAllowUnknown(args);
  }

  SplitEvaluator* GetHostClone() const override {
    auto r = new RidgePenalty();
    r->params_ = this->params_;

    return r;
  }

  bst_float ComputeSplitScore(bst_uint nodeid,
                             bst_uint featureid,
                             const GradStats& left,
                             const GradStats& right) const override {
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

 private:
  RidgePenaltyParams params_;
};

XGBOOST_REGISTER_SPLIT_EVALUATOR(RidgePenalty, "ridge")
.describe("Use an L2 penalty term for the weights and a cost per leaf node")
.set_body([]() {
    return new RidgePenalty();
  });

/*! \brief Encapsulates the parameters required by the MonotonicConstraint
        split evaluator
*/
struct MonotonicConstraintParams
    : public dmlc::Parameter<MonotonicConstraintParams> {
  std::vector<bst_int> monotone_constraints;
  float reg_lambda;
  float reg_gamma;

  DMLC_DECLARE_PARAMETER(MonotonicConstraintParams) {
    DMLC_DECLARE_FIELD(reg_lambda)
      .set_lower_bound(0.0)
      .set_default(1.0)
      .describe("L2 regularization on leaf weight");
    DMLC_DECLARE_FIELD(reg_gamma)
      .set_lower_bound(0.0f)
      .set_default(0.0f)
      .describe("Cost incurred by adding a new leaf node to the tree");
    DMLC_DECLARE_FIELD(monotone_constraints)
      .set_default(std::vector<bst_int>())
      .describe("Constraint of variable monotonicity");
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_gamma, gamma);
  }
};

DMLC_REGISTER_PARAMETER(MonotonicConstraintParams);

/*! \brief Enforces that the tree is monotonically increasing/decreasing with respect to a user specified set of
      features.
*/
class MonotonicConstraint final : public SplitEvaluator {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args)
      override {
    params_.InitAllowUnknown(args);
    Reset();
  }

  void Reset() override {
    lower_.resize(1, -std::numeric_limits<bst_float>::max());
    upper_.resize(1, std::numeric_limits<bst_float>::max());
  }

  SplitEvaluator* GetHostClone() const override {
    if (params_.monotone_constraints.size() == 0) {
      // No monotone constraints specified, make a RidgePenalty evaluator
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
      auto c = new MonotonicConstraint();
      c->params_ = this->params_;
      c->Reset();
      return c;
    }
  }

  bst_float ComputeSplitScore(bst_uint nodeid,
                             bst_uint featureid,
                             const GradStats& left,
                             const GradStats& right) const override {
    bst_float infinity = std::numeric_limits<bst_float>::infinity();
    bst_int constraint = GetConstraint(featureid);

    bst_float score = ComputeScore(nodeid, left) + ComputeScore(nodeid, right);
    bst_float leftweight = ComputeWeight(nodeid, left);
    bst_float rightweight = ComputeWeight(nodeid, right);

    if (constraint == 0) {
      return score;
    } else if (constraint > 0) {
      return leftweight <= rightweight ? score : -infinity;
    } else {
      return leftweight >= rightweight ? score : -infinity;
    }
  }

  bst_float ComputeScore(bst_uint parentID, const GradStats& stats)
      const override {
    bst_float w = ComputeWeight(parentID, stats);

    return -(2.0 * stats.sum_grad * w + (stats.sum_hess + params_.reg_lambda)
        * w * w);
  }

  bst_float ComputeWeight(bst_uint parentID, const GradStats& stats)
      const override {
    bst_float weight = -stats.sum_grad / (stats.sum_hess + params_.reg_lambda);

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
.set_body([]() {
    return new MonotonicConstraint();
  });

}  // namespace tree
}  // namespace xgboost
