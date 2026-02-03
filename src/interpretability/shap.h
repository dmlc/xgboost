/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#pragma once

#include <vector>  // for vector

#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for DMatrix, MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::gbm {
struct GBTreeModel;
}  // namespace xgboost::gbm

namespace xgboost::interpretability {
void ShapValues(Context const *ctx, DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                gbm::GBTreeModel const &model, bst_tree_t tree_end,
                std::vector<float> const *tree_weights, int condition, unsigned condition_feature);

void ApproxFeatureImportance(Context const *ctx, DMatrix *p_fmat,
                             HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                             bst_tree_t tree_end, std::vector<float> const *tree_weights);

void ShapInteractionValues(Context const *ctx, DMatrix *p_fmat,
                           HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                           bst_tree_t tree_end, std::vector<float> const *tree_weights,
                           bool approximate);
}  // namespace xgboost::interpretability
