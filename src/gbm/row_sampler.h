/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <memory>  // for unique_ptr

#include "../tree/param.h"
#include "xgboost/context.h"
#include "xgboost/gradient.h"

namespace xgboost::gbm {
/** @brief Draw independent samples of one boosting round's unchanged gradients.
 *
 * Preparation is shared across parallel trees. Each call draws a fresh sample and
 * applies matching weights to split and optional leaf-value gradients. The input
 * must outlive this sampler and remain unchanged.
 */
class RowSampler {
 public:
  virtual void Sample(Context const* ctx, GradientContainer* out) = 0;
  virtual ~RowSampler() = default;
};

struct MakeRowSamplerKernel {
  using Signature = std::unique_ptr<RowSampler>(Context const*, tree::TrainParam const&,
                                                GradientContainer const&);
};
}  // namespace xgboost::gbm
