#ifndef XGBOOST_LEARNER_EVALUATION_H_
#define XGBOOST_LEARNER_EVALUATION_H_
/*!
 * \file evaluation.h
 * \brief interface of evaluation function supported in xgboost
 * \author Tianqi Chen, Kailong Chen
 */
#include <string>
#include <vector>
#include <cstdio>
#include "../utils/utils.h"
#include "./dmatrix.h"

namespace xgboost {
namespace learner {
/*! \brief evaluator that evaluates the loss metrics */
struct IEvaluator{
  /*!
   * \brief evaluate a specific metric
   * \param preds prediction
   * \param info information, including label etc.
   */
  virtual float Eval(const std::vector<float> &preds,
                     const MetaInfo &info) const = 0;
  /*! \return name of metric */
  virtual const char *Name(void) const = 0;
  /*! \brief virtual destructor */
  virtual ~IEvaluator(void) {}
};
}  // namespace learner
}  // namespace xgboost

// include implementations of evaluation functions
#include "evaluation-inl.hpp"
// factory function
namespace xgboost {
namespace learner {
inline IEvaluator* CreateEvaluator(const char *name) {
  if (!strcmp(name, "rmse")) return new EvalRMSE();
  if (!strcmp(name, "error")) return new EvalError();
  if (!strcmp(name, "merror")) return new EvalMatchError();
  if (!strcmp(name, "logloss")) return new EvalLogLoss();
  if (!strcmp(name, "auc")) return new EvalAuc();
  if (!strncmp(name, "ams@", 4)) return new EvalAMS(name);
  if (!strncmp(name, "pre@", 4)) return new EvalPrecision(name);
  if (!strncmp(name, "pratio@", 7)) return new EvalPrecisionRatio(name);
  if (!strncmp(name, "map", 3)) return new EvalMAP(name);
  if (!strncmp(name, "ndcg", 4)) return new EvalNDCG(name);
  if (!strncmp(name, "ct-", 3)) return new EvalCTest(CreateEvaluator(name+3), name);

  utils::Error("unknown evaluation metric type: %s", name);
  return NULL;
}

/*! \brief a set of evaluators */
class EvalSet{
 public:
  inline void AddEval(const char *name) {
    for (size_t i = 0; i < evals_.size(); ++i) {
      if (!strcmp(name, evals_[i]->Name())) return;
    }
    evals_.push_back(CreateEvaluator(name));
  }
  ~EvalSet(void) {
    for (size_t i = 0; i < evals_.size(); ++i) {
      delete evals_[i];
    }
  }
  inline std::string Eval(const char *evname,
                          const std::vector<float> &preds,
                          const MetaInfo &info) const {
    std::string result = "";
    for (size_t i = 0; i < evals_.size(); ++i) {
      float res = evals_[i]->Eval(preds, info);
      char tmp[1024];
      utils::SPrintf(tmp, sizeof(tmp), "\t%s-%s:%f", evname, evals_[i]->Name(), res);
      result += tmp;
    }
    return result;
  }

 private:
  std::vector<const IEvaluator*> evals_;
};
}  // namespace learner
}  // namespace xgboost
#endif  // XGBOOST_LEARNER_EVALUATION_H_
