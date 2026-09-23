/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#ifndef XGBOOST_TASK_H_
#define XGBOOST_TASK_H_

#include <cstdint>  // for uint8_t

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
  enum Task : std::uint8_t {
    kRegression = 0,
    kBinary = 1,
    kClassification = 2,
    kSurvival = 3,
    kRanking = 4,
    kOther = 5,
  } task;
  // Does the objective have constant hessian value?
  bool const_hess{false};
  /**
   * @brief Can the objective produce an exact dense Hessian?
   *
   * This states a property of the objective's mathematics only: that it implements
   * ObjFunction::GetGradientAndExactHessian. Whether a particular run can actually use
   * that producer is a separate question -- it depends on the device, the tree method and
   * the training configuration -- and is answered by the training path, not here.
   */
  bool exact_hess{false};

  ObjInfo(Task t) : task{t} {}  // NOLINT
  ObjInfo(Task t, bool khess) : task{t}, const_hess{khess} {}
  ObjInfo(Task t, bool khess, bool kexact) : task{t}, const_hess{khess}, exact_hess{kexact} {}
};
}  // namespace xgboost
#endif  // XGBOOST_TASK_H_
