/*!
 * Copyright 2021-2022 by XGBoost Contributors
 */
#ifndef XGBOOST_TASK_H_
#define XGBOOST_TASK_H_

#include <xgboost/base.h>

#include <cinttypes>

namespace xgboost {
/*!
 * \brief A struct returned by objective, which determines task at hand.  The struct is
 *        not used by any algorithm yet, only for future development like categorical
 *        split.
 *
 * The task field is useful for tree split finding, also for some metrics like auc.
 * Lastly, knowing whether hessian is constant can allow some optimizations like skipping
 * the quantile sketching.
 *
 * This struct should not be serialized since it can be recovered from objective function,
 * hence it doesn't need to be stable.
 */
struct ObjInfo {
  // What kind of problem are we trying to solve
  enum Task : uint8_t {
    kRegression = 0,
    kBinary = 1,
    kClassification = 2,
    kSurvival = 3,
    kRanking = 4,
    kOther = 5,
  } task;
  // Does the objective have constant hessian value?
  bool const_hess{false};

  explicit ObjInfo(Task t) : task{t} {}
  ObjInfo(Task t, bool khess) : task{t}, const_hess{khess} {}

  XGBOOST_DEVICE bool UseOneHot() const {
    return (task != ObjInfo::kRegression && task != ObjInfo::kBinary);
  }
};
}  // namespace xgboost
#endif  // XGBOOST_TASK_H_
