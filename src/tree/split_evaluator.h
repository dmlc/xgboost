/*!
 * Copyright 2018 by Contributors
 * \file split_evaluator.h
 * \brief Used for implementing a loss term specific to decision trees. Useful for custom regularisation.
 * \author Henry Gouk
 */

#ifndef XGBOOST_SPLIT_EVALUATOR_H_
#define XGBOOST_SPLIT_EVALUATOR_H_

#include <dmlc/registry.h>
#include <xgboost/base.h>
#include <functional>
#include <string>
#include <vector>

namespace xgboost {
namespace tree {

// Should GradStats be in this header, rather than param.h?
struct GradStats;

class SplitEvaluator {
 public:
  // Factory method for constructing new SplitEvaluators
  static SplitEvaluator* Create(const std::string& name);

  virtual ~SplitEvaluator();

  // Used to initialise any regularisation hyperparameters provided by the user
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& args);

  // Resets the SplitEvaluator to the state it was in immediately after the Init method was called
  virtual void Reset();

  // This will create a clone of the SplitEvaluator in host memory
  virtual SplitEvaluator* GetHostClone() const = 0;

  // Convenience method that calls ComputeLoss for two different GradStats instances
  virtual bst_float ComputeSplitLoss(bst_uint parentID,
                                     bst_uint featureID,
                                     const GradStats& left,
                                     const GradStats& right) const;

  // Compute the loss for a node with the given stats
  virtual bst_float ComputeLoss(bst_uint parentID, const GradStats& stats) const = 0;

  // Compute the weight for a node with the given stats
  virtual bst_float ComputeWeight(bst_uint parentID, const GradStats& stats) const = 0;
};

struct SplitEvaluatorReg
    : public dmlc::FunctionRegEntryBase<SplitEvaluatorReg,
                                        std::function<SplitEvaluator* ()> > {};

/*!
 * \brief Macro to register tree split evaluator.
 *
 * \code
 * // example of registering a split evaluator
 * XGBOOST_REGISTER_SPLIT_EVALUATOR(SplitEval, "splitEval")
 * .describe("Some split evaluator")
 * .set_body([]() {
 *     return new SplitEval();
 *   });
 * \endcode
 */
#define XGBOOST_REGISTER_SPLIT_EVALUATOR(UniqueID, Name) \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::tree::SplitEvaluatorReg& \
  __make_ ## SplitEvaluatorReg ## _ ## UniqueID ## __ = \
      ::dmlc::Registry< ::xgboost::tree::SplitEvaluatorReg>::Get()->__REGISTER__(Name)

} // tree
} // xgboost

#endif // XGBOOST_SPLIT_EVALUATOR_H_