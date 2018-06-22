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

std::string CreateBigTestData(size_t n_entries);

void CheckObjFunction(xgboost::ObjFunction * obj,
                      std::vector<xgboost::bst_float> preds,
                      std::vector<xgboost::bst_float> labels,
                      std::vector<xgboost::bst_float> weights,
                      std::vector<xgboost::bst_float> out_grad,
                      std::vector<xgboost::bst_float> out_hess);

void CheckRankingObjFunction(xgboost::ObjFunction * obj,
                      std::vector<xgboost::bst_float> preds,
                      std::vector<xgboost::bst_float> labels,
                      std::vector<xgboost::bst_float> weights,
                      std::vector<xgboost::bst_uint> groups,
                      std::vector<xgboost::bst_float> out_grad,
                      std::vector<xgboost::bst_float> out_hess);

xgboost::bst_float GetMetricEval(
  xgboost::Metric * metric,
  std::vector<xgboost::bst_float> preds,
  std::vector<xgboost::bst_float> labels,
  std::vector<xgboost::bst_float> weights = std::vector<xgboost::bst_float> ());

/**
 * \fn  std::shared_ptr<xgboost::DMatrix> CreateDMatrix(int rows, int columns, float sparsity, int seed);
 *
 * \brief Creates dmatrix with uniform random data between 0-1.
 *
 * \param rows      The rows.
 * \param columns   The columns.
 * \param sparsity  The sparsity.
 * \param seed      The seed.
 *
 * \return  The new d matrix.
 */

std::shared_ptr<xgboost::DMatrix> CreateDMatrix(int rows, int columns,
                                                float sparsity, int seed = 0);
#endif
