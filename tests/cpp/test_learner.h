/*!
 * Copyright 2018 by Contributors
 * \file test_learner.h
 * \brief Hook to access implementation class of Learner
 * \author Hyunsu Philip Cho
 */

#ifndef XGBOOST_TESTS_CPP_TEST_LEARNER_H_
#define XGBOOST_TESTS_CPP_TEST_LEARNER_H_

#include <string>

namespace xgboost {
class LearnerTestHook {
 private:
  virtual std::string GetUpdaterSequence(void) const = 0;
  // allow friend access to C++ tests for Learner
  friend class LearnerTestHookAdapter;
};
}  // namespace xgboost

#endif  // XGBOOST_TESTS_CPP_TEST_LEARNER_H_
