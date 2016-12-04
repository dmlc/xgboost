#ifndef XGBOOST_TESTS_CPP_HELPERS_H_
#define XGBOOST_TESTS_CPP_HELPERS_H_

#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include <gtest/gtest.h>

#include <xgboost/base.h>
#include <xgboost/objective.h>
#include <xgboost/metric.h>

std::string TempFileName();

bool FileExists(const std::string name);

long GetFileSize(const std::string filename);

std::string CreateSimpleTestData();

void CheckObjFunction(xgboost::ObjFunction * obj,
                      std::vector<xgboost::bst_float> preds,
                      std::vector<xgboost::bst_float> labels,
                      std::vector<xgboost::bst_float> weights,
                      std::vector<xgboost::bst_float> out_grad,
                      std::vector<xgboost::bst_float> out_hess);

xgboost::bst_float GetMetricEval(
  xgboost::Metric * metric,
  std::vector<xgboost::bst_float> preds,
  std::vector<xgboost::bst_float> labels,
  std::vector<xgboost::bst_float> weights = std::vector<xgboost::bst_float> ());

#endif
