/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_TASK_H_
#define XGBOOST_TASK_H_

#include <cinttypes>

namespace xgboost {
/*!
 * \brief A struct returned by objective, which determines task at hand.  The struct is
 *        not used by any algorithm yet, only for future development like categorical
 *        split.
 *
 * The task field is useful for tree split finding, also for some metrics like auc.  While
 * const_hess is useful for algorithms like adaptive tree where one needs to update the
 * leaf value after building the tree.  Lastly, knowing whether hessian is constant can
 * allow some optimizations like skipping the quantile sketching.
 *
 * This struct should not be serialized since it can be recovered from objective function,
 * hence it doesn't need to be stable.
 */
struct ObjInfo {
  // What kind of problem are we trying to solve
  enum : uint8_t {
    kRegression = 0,
    kBinary = 1,
    kClassification = 2,
    kSurvival = 3,
    kRanking = 4,
    kOther = 5,
  } task;
  // Does the objective have constant hessian value?
  bool const_hess{false};
};
}  // namespace xgboost
#endif  // XGBOOST_TASK_H_
