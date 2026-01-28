/**
 * Copyright 2026, XGBoost Contributors
 */
#include <cstddef>  // for size_t

#include "xgboost/base.h"    // for GradientPair
#include "xgboost/linalg.h"  // for MatrixView
#include "xgboost/span.h"    // for Span

namespace xgboost::tree::cpu_impl {
std::size_t CalculateThresholdIndex(common::Span<float> threshold, float subsample) {}

void Sample(linalg::MatrixView<GradientPair const> gpairs) {}
}  // namespace xgboost::tree::cpu_impl
