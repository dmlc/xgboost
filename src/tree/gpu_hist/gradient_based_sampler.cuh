/*!
 * Copyright 2019 by XGBoost Contributors
 */
#pragma once
#include <xgboost/data.h>

namespace xgboost {
namespace tree {

struct GradientBasedSample {
  /*!\brief The sample rows in ELLPACK format. */
  EllpackPageImpl* page;
  /*!\brief Rescaled gradient pairs for the sampled rows. */
  HostDeviceVector<GradientPair>* gpair;
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
  GradientBasedSample Sample(HostDeviceVector<GradientPair>* gpair,
                             DMatrix* dmat,
                             BatchParam batch_param);

  void Sample(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat, size_t sample_rows);
};
};  // namespace tree
};  // namespace xgboost
