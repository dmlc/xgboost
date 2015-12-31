/*!
 * Copyright 2015 by Contributors
 * \file metric_set.h
 * \brief additional math utils
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_METRIC_SET_H_
#define XGBOOST_COMMON_METRIC_SET_H_

#include <vector>
#include <string>

namespace xgboost {
namespace common {

/*! \brief helper util to create a set of metrics */
class MetricSet {
  inline void AddEval(const char *name) {
    using namespace std;
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
                          const MetaInfo &info,
                          bool distributed = false) {
    std::string result = "";
    for (size_t i = 0; i < evals_.size(); ++i) {
      float res = evals_[i]->Eval(preds, info, distributed);
      char tmp[1024];
      utils::SPrintf(tmp, sizeof(tmp), "\t%s-%s:%f", evname, evals_[i]->Name(), res);
      result += tmp;
    }
    return result;
  }
  inline size_t Size(void) const {
    return evals_.size();
  }

 private:
  std::vector<const IEvaluator*> evals_;
};

}  // namespace common
}  // namespace xgboost
#endif   // XGBOOST_COMMON_METRIC_SET_H_
