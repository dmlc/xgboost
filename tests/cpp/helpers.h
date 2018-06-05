#ifndef XGBOOST_TESTS_CPP_HELPERS_H_
#define XGBOOST_TESTS_CPP_HELPERS_H_

#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <xgboost/base.h>
#include <xgboost/metric.h>
#include <xgboost/objective.h>
#include "../../src/gbm/gbtree_model.h"

std::string TempFileName();

bool FileExists(const std::string name);

long GetFileSize(const std::string filename);

std::string CreateSimpleTestData();

void CheckObjFunction(xgboost::ObjFunction* obj,
                      std::vector<xgboost::bst_float> preds,
                      std::vector<xgboost::bst_float> labels,
                      std::vector<xgboost::bst_float> weights,
                      std::vector<xgboost::bst_float> out_grad,
                      std::vector<xgboost::bst_float> out_hess);

xgboost::bst_float GetMetricEval(xgboost::Metric* metric,
                                 std::vector<xgboost::bst_float> preds,
                                 std::vector<xgboost::bst_float> labels,
                                 std::vector<xgboost::bst_float> weights =
                                     std::vector<xgboost::bst_float>());

/**
 * \fn  std::shared_ptr<xgboost::DMatrix> CreateDMatrix(int rows, int columns,
 * float sparsity, int seed);
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
/**
 * \fn  gbm::GBTreeModel CreateModel(int num_trees, int num_group,
 * std::vector<float> leaf_weights, float base_margin)
 *
 * \brief Creates a gradient boosting model for testing. Each tree has a single
 * leaf node. Can create multiple groups with a different leaf weight predicted
 * for each group.
 *
 * \param num_trees     Number of trees.
 * \param num_group     Number of groups.
 * \param leaf_weights  The leaf weights. One for each group.
 * \param base_margin   The base margin.
 *
 * \return  The new model.
 */

xgboost::gbm::GBTreeModel CreateModel(int num_trees, int num_group,
                                      std::vector<float> leaf_weights,
                                      float base_margin);
#endif
