/*!
 * Copyright 2018 by Contributors
 * \file split_evaluator.cc
 * \brief Contains implementations of different split evaluators.
 */
#include <algorithm>
#include <limits>
#include <utility>
#include <dmlc/registry.h>
#include "split_evaluator.h"
#include "param.h"
#include "../common/common.h"
#include "../common/host_device_vector.h"

#define ROOT_PARENT_ID (-1 & ((1U << 31) - 1))

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::tree::SplitEvaluatorReg);
} // namespace dmlc

namespace xgboost {
namespace tree {

SplitEvaluator* SplitEvaluator::Create(const std::string& name) {
  auto* e = ::dmlc::Registry< ::xgboost::tree::SplitEvaluatorReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown SplitEvaluator " << name;
  }
  return (e->body)();
}

// Default implementations of some virtual methods that don't need to do anything by default
SplitEvaluator::~SplitEvaluator() {}
void SplitEvaluator::Init(const std::vector<std::pair<std::string, std::string> >& args) {}
void SplitEvaluator::Reset() {}
void SplitEvaluator::AddSplit(bst_uint nodeID,
                              bst_uint leftID,
                              bst_uint rightID,
                              bst_uint featureID,
                              bst_float leftWeight,
                              bst_float rightWeight) {}

/*! \brief Encapsulates the parameters required by the RidgePenalty split evaluator */
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
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    m_params.InitAllowUnknown(args);
  }

  SplitEvaluator* GetHostClone() const override {
    RidgePenalty* r = new RidgePenalty();
    r->m_params = this->m_params;

    return r;
  }

  bst_float ComputeSplitScore(bst_uint nodeID,
                             bst_uint featureID,
                             const GradStats& left,
                             const GradStats& right) const {
    // nodeID parameter is not used by the SpitEvaluator, so just call other members with 0 to prevent code duplication
    return ComputeScore(0, left) + ComputeScore(0, right);
  }

  bst_float ComputeScore(bst_uint nodeID, const GradStats& stats) const override {
    return (stats.sum_grad * stats.sum_grad) / (stats.sum_hess + m_params.reg_lambda) - m_params.reg_gamma;
  }

  bst_float ComputeWeight(bst_uint nodeID, const GradStats& stats) const override {
    return -stats.sum_grad / (stats.sum_hess + m_params.reg_lambda);
  }

 private:
  RidgePenaltyParams m_params;
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
    m_params.InitAllowUnknown(args);
    Reset();
  }

  void Reset() override {
    m_lower.resize(1, -std::numeric_limits<bst_float>::max());
    m_upper.resize(1, std::numeric_limits<bst_float>::max());
  }

  SplitEvaluator* GetHostClone() const override {
    MonotonicConstraint* c = new MonotonicConstraint();
    c->m_params = this->m_params;
    c->Reset();

    return c;
  }

  bst_float ComputeSplitScore(bst_uint nodeID,
                             bst_uint featureID,
                             const GradStats& left,
                             const GradStats& right) const override {
    bst_float infinity = std::numeric_limits<bst_float>::infinity();
    bst_int constraint = getConstraint(featureID);

    bst_float score = ComputeScore(nodeID, left) + ComputeScore(nodeID, right);
    bst_float leftWeight = ComputeWeight(nodeID, left);
    bst_float rightWeight = ComputeWeight(nodeID, right);

    if (constraint == 0) {
      return score;
    } else if (constraint > 0) {
      return leftWeight <= rightWeight ? score : -infinity;
    } else {
      return leftWeight >= rightWeight ? score : -infinity;
    }
  }

  bst_float ComputeScore(bst_uint parentID, const GradStats& stats) const override {
    bst_float w = ComputeWeight(parentID, stats);

    return -(2.0 * stats.sum_grad * w + (stats.sum_hess + m_params.reg_lambda) * w * w);
  }

  bst_float ComputeWeight(bst_uint parentID, const GradStats& stats) const override {
    bst_float weight = -stats.sum_grad / (stats.sum_hess + m_params.reg_lambda);

    if (parentID == ROOT_PARENT_ID) {
      // This is the root node
      return weight;
    } else if (weight < m_lower.at(parentID)) {
      return m_lower.at(parentID);
    } else if (weight > m_upper.at(parentID)) {
      return m_upper.at(parentID);
    } else {
      return weight;
    }
  }

  void AddSplit(bst_uint nodeID,
                bst_uint leftID,
                bst_uint rightID,
                bst_uint featureID,
                bst_float leftWeight,
                bst_float rightWeight) override {
    bst_uint newSize = std::max(leftID, rightID) + 1;
    m_lower.resize(newSize);
    m_upper.resize(newSize);
    bst_int constraint = getConstraint(featureID);

    bst_float mid = (leftWeight + rightWeight) / 2;
    CHECK(!std::isnan(mid));

    m_upper[leftID] = m_upper.at(nodeID);
    m_upper[rightID] = m_upper.at(nodeID);
    m_lower[leftID] = m_lower.at(nodeID);
    m_lower[rightID] = m_lower.at(nodeID);

    if (constraint < 0) {
      m_lower[leftID] = mid;
      m_upper[rightID] = mid;
    } else if (constraint > 0) {
      m_upper[leftID] = mid;
      m_lower[rightID] = mid;
    }
  }

 private:
  MonotonicConstraintParams m_params;
  std::vector<bst_float> m_lower;
  std::vector<bst_float> m_upper;

  inline bst_int getConstraint(bst_uint featureID) const {
    if (featureID < m_params.monotone_constraints.size()) {
      return m_params.monotone_constraints[featureID];
    } else {
      return 0;
    }
  }
};

XGBOOST_REGISTER_SPLIT_EVALUATOR(MonotonicConstraint, "monotonic")
.describe("Enforces that the tree is monotonically increasing/decreasing w.r.t. specified features")
.set_body([]() {
    return new MonotonicConstraint();
  });

} // tree
} // xgboost
