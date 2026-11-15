/**
 * Copyright 2026, XGBoost Contributors
 * \file radix_select.h
 * \brief Float32 radix-selection kernel for objective intercepts.
 */
#ifndef XGBOOST_OBJECTIVE_RADIX_SELECT_H_
#define XGBOOST_OBJECTIVE_RADIX_SELECT_H_

#include "xgboost/context.h"             // for Context
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix, Vector

namespace xgboost::obj {
struct RadixSelectKernel {
  using Signature = void(Context const*, linalg::Matrix<float> const&,
                         HostDeviceVector<float> const&, HostDeviceVector<float> const&,
                         bst_target_t, linalg::Vector<float>*);
};

/**
 * @brief Select weighted float32 quantiles without sorting.
 *
 * Absolute-error and quantile objectives need exact weighted quantiles to initialize their
 * intercepts. Sorting all labels would require O(n log n) work and is inconvenient on GPUs and
 * across distributed workers. Since the labels are float32, their ordered representation has
 * only 32 bits. We can recover the selected value directly with four passes over the data.
 *
 * Each pass considers the next 8 bits, from most to least significant. It builds a weighted
 * 256-bin histogram for values matching the prefix selected by previous passes, sums the
 * histogram across workers, and selects the bin containing the requested cumulative weight. The
 * selected bin extends the prefix by 8 bits and the weight of preceding bins is subtracted from
 * the rank. After four passes the 32-bit prefix identifies the exact selected float.
 *
 * The result contains one quantile for each (input column, alpha) pair, with alpha varying
 * fastest. The input is never sorted or modified. Other devices use the registered CPU kernel
 * fallback. A globally zero total weight produces zero.
 *
 * n_targets is the agreed output count, supplied by the caller without further synchronization.
 * It is independent of the local label shape: for quantile regression, the configured alphas
 * determine the output count even on a worker with a 0-by-0 label matrix. That worker must still
 * allocate the same histograms and participate in all four passes.
 */
void RadixSelect(Context const* ctx, linalg::Matrix<float> const& values,
                 HostDeviceVector<float> const& weights, HostDeviceVector<float> const& alphas,
                 bst_target_t n_targets, linalg::Vector<float>* out);
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_RADIX_SELECT_H_
