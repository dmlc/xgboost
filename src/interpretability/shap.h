/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#pragma once

#include <vector>  // for vector

#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for DMatrix, MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/span.h"                // for Span

namespace xgboost::gbm {
struct GBTreeModel;
}  // namespace xgboost::gbm

namespace xgboost::interpretability {
namespace cpu_impl {
void ShapValues(Context const *ctx, DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                gbm::GBTreeModel const &model, bst_tree_t tree_end,
                common::Span<float const> tree_weights, int condition, unsigned condition_feature);

void ApproxFeatureImportance(Context const *ctx, DMatrix *p_fmat,
                             HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                             bst_tree_t tree_end, common::Span<float const> tree_weights);

void ShapInteractionValues(Context const *ctx, DMatrix *p_fmat,
                           HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                           bst_tree_t tree_end, common::Span<float const> tree_weights,
                           bool approximate);
}  // namespace cpu_impl

#if defined(XGBOOST_USE_CUDA)
namespace cuda_impl {
void ShapValues(Context const *ctx, DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                gbm::GBTreeModel const &model, bst_tree_t tree_end,
                common::Span<float const> tree_weights, int condition, unsigned condition_feature);
void ApproxFeatureImportance(Context const *ctx, DMatrix *p_fmat,
                             HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                             bst_tree_t tree_end, common::Span<float const> tree_weights);
void ShapInteractionValues(Context const *ctx, DMatrix *p_fmat,
                           HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                           bst_tree_t tree_end, common::Span<float const> tree_weights,
                           bool approximate);
}  // namespace cuda_impl
#endif  // defined(XGBOOST_USE_CUDA)

inline void ShapValues(Context const *ctx, DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                       gbm::GBTreeModel const &model, bst_tree_t tree_end,
                       common::Span<float const> tree_weights, int condition,
                       unsigned condition_feature) {
#if defined(XGBOOST_USE_CUDA)
  if (ctx->IsCUDA()) {
    cuda_impl::ShapValues(ctx, p_fmat, out_contribs, model, tree_end, tree_weights, condition,
                          condition_feature);
    return;
  }
#endif  // defined(XGBOOST_USE_CUDA)
  cpu_impl::ShapValues(ctx, p_fmat, out_contribs, model, tree_end, tree_weights, condition,
                       condition_feature);
}

inline void ApproxFeatureImportance(Context const *ctx, DMatrix *p_fmat,
                                    HostDeviceVector<float> *out_contribs,
                                    gbm::GBTreeModel const &model, bst_tree_t tree_end,
                                    common::Span<float const> tree_weights) {
#if defined(XGBOOST_USE_CUDA)
  if (ctx->IsCUDA()) {
    cuda_impl::ApproxFeatureImportance(ctx, p_fmat, out_contribs, model, tree_end, tree_weights);
    return;
  }
#endif  // defined(XGBOOST_USE_CUDA)
  cpu_impl::ApproxFeatureImportance(ctx, p_fmat, out_contribs, model, tree_end, tree_weights);
}

inline void ShapInteractionValues(Context const *ctx, DMatrix *p_fmat,
                                  HostDeviceVector<float> *out_contribs,
                                  gbm::GBTreeModel const &model, bst_tree_t tree_end,
                                  common::Span<float const> tree_weights, bool approximate) {
#if defined(XGBOOST_USE_CUDA)
  if (ctx->IsCUDA()) {
    cuda_impl::ShapInteractionValues(ctx, p_fmat, out_contribs, model, tree_end, tree_weights,
                                     approximate);
    return;
  }
#endif  // defined(XGBOOST_USE_CUDA)
  cpu_impl::ShapInteractionValues(ctx, p_fmat, out_contribs, model, tree_end, tree_weights,
                                  approximate);
}
}  // namespace xgboost::interpretability
