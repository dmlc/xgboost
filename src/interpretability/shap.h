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
void ShapValuesCPU(Context const *ctx, DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                   gbm::GBTreeModel const &model, bst_tree_t tree_end,
                   std::vector<float> const *tree_weights, int condition,
                   unsigned condition_feature);

void ApproxFeatureImportanceCPU(Context const *ctx, DMatrix *p_fmat,
                                HostDeviceVector<float> *out_contribs,
                                gbm::GBTreeModel const &model, bst_tree_t tree_end,
                                std::vector<float> const *tree_weights);

void ShapInteractionValuesCPU(Context const *ctx, DMatrix *p_fmat,
                              HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                              bst_tree_t tree_end, std::vector<float> const *tree_weights,
                              bool approximate);

#if defined(XGBOOST_USE_CUDA)
void ShapValuesCUDA(Context const *ctx, DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                    gbm::GBTreeModel const &model, bst_tree_t tree_end,
                    std::vector<float> const *tree_weights, int condition,
                    unsigned condition_feature);
void ApproxFeatureImportanceCUDA(Context const *ctx, DMatrix *p_fmat,
                                 HostDeviceVector<float> *out_contribs,
                                 gbm::GBTreeModel const &model, bst_tree_t tree_end,
                                 std::vector<float> const *tree_weights);
void ShapInteractionValuesCUDA(Context const *ctx, DMatrix *p_fmat,
                               HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                               bst_tree_t tree_end, std::vector<float> const *tree_weights,
                               bool approximate);
#endif  // defined(XGBOOST_USE_CUDA)

inline void ShapValues(Context const *ctx, DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                       gbm::GBTreeModel const &model, bst_tree_t tree_end,
                       std::vector<float> const *tree_weights, int condition,
                       unsigned condition_feature) {
#if defined(XGBOOST_USE_CUDA)
  if (ctx->IsCUDA()) {
    ShapValuesCUDA(ctx, p_fmat, out_contribs, model, tree_end, tree_weights, condition,
                   condition_feature);
    return;
  }
#endif  // defined(XGBOOST_USE_CUDA)
  ShapValuesCPU(ctx, p_fmat, out_contribs, model, tree_end, tree_weights, condition,
                condition_feature);
}

inline void ApproxFeatureImportance(Context const *ctx, DMatrix *p_fmat,
                                    HostDeviceVector<float> *out_contribs,
                                    gbm::GBTreeModel const &model, bst_tree_t tree_end,
                                    std::vector<float> const *tree_weights) {
#if defined(XGBOOST_USE_CUDA)
  if (ctx->IsCUDA()) {
    ApproxFeatureImportanceCUDA(ctx, p_fmat, out_contribs, model, tree_end, tree_weights);
    return;
  }
#endif  // defined(XGBOOST_USE_CUDA)
  ApproxFeatureImportanceCPU(ctx, p_fmat, out_contribs, model, tree_end, tree_weights);
}

inline void ShapInteractionValues(Context const *ctx, DMatrix *p_fmat,
                                  HostDeviceVector<float> *out_contribs,
                                  gbm::GBTreeModel const &model, bst_tree_t tree_end,
                                  std::vector<float> const *tree_weights, bool approximate) {
#if defined(XGBOOST_USE_CUDA)
  if (ctx->IsCUDA()) {
    ShapInteractionValuesCUDA(ctx, p_fmat, out_contribs, model, tree_end, tree_weights,
                              approximate);
    return;
  }
#endif  // defined(XGBOOST_USE_CUDA)
  ShapInteractionValuesCPU(ctx, p_fmat, out_contribs, model, tree_end, tree_weights, approximate);
}
}  // namespace xgboost::interpretability
