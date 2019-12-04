/*!
 * Copyright 2019 by XGBoost Contributors
 */
#pragma once
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/span.h>

#include "../../common/device_helpers.cuh"
#include "../../data/ellpack_page.cuh"

namespace xgboost {
namespace tree {

struct GradientBasedSample {
  /*!\brief Number of sampled rows. */
  size_t sample_rows;
  /*!\brief Sampled rows in ELLPACK format. */
  EllpackPageImpl* page;
  /*!\brief Rescaled gradient pairs for the sampled rows. */
  common::Span<GradientPair> gpair;
};

/*! \brief Draw a sample of rows from a DMatrix.
 *
 * Use Poisson sampling to draw a probability proportional to size (pps) sample of rows from a
 * DMatrix, where "size" is the absolute value of the gradient.
 *
 * \see Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017).
 * Lightgbm: A highly efficient gradient boosting decision tree. In Advances in Neural Information
 * Processing Systems (pp. 3146-3154).
 * \see Zhu, R. (2016). Gradient-based sampling: An adaptive importance sampling for least-squares.
 * In Advances in Neural Information Processing Systems (pp. 406-414).
 * \see Ohlsson, E. (1998). Sequential poisson sampling. Journal of official Statistics, 14(2), 149.
 */
class GradientBasedSampler {
 public:
  explicit GradientBasedSampler(BatchParam batch_param,
                                EllpackInfo info,
                                size_t n_rows,
                                size_t sample_rows = 0);

  /*! \brief Returns the max number of rows that can fit in available GPU memory. */
  size_t MaxSampleRows();

  GradientBasedSample Sample(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat);

 private:
  common::Monitor monitor_;
  dh::BulkAllocator ba_;
  BatchParam batch_param_;
  EllpackInfo info_;
  bool is_sampling_;
  size_t sample_rows_;
  std::unique_ptr<EllpackPageImpl> page_;
  common::Span<GradientPair> gpair_;
};
};  // namespace tree
};  // namespace xgboost
