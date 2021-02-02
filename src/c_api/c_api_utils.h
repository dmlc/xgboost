/*!
 * Copyright (c) 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_C_API_C_API_UTILS_H_
#define XGBOOST_C_API_C_API_UTILS_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "xgboost/logging.h"
#include "xgboost/learner.h"

namespace xgboost {
/* \brief Determine the output shape of prediction.
 *
 * \param strict_shape Whether should we reshape the output with consideration of groups
 *                     and forest.
 * \param type         Prediction type
 * \param rows         Input samples
 * \param cols         Input features
 * \param chunksize    Total elements of output / rows
 * \param groups       Number of output groups from Learner
 * \param rounds       end_iteration - beg_iteration
 * \param out_shape    Output shape
 * \param out_dim      Output dimension
 */
inline void CalcPredictShape(bool strict_shape, PredictionType type, size_t rows, size_t cols,
                             size_t chunksize, size_t groups, size_t rounds,
                             std::vector<bst_ulong> *out_shape,
                             xgboost::bst_ulong *out_dim) {
  auto &shape = *out_shape;
  if (type == PredictionType::kMargin && rows != 0) {
    // When kValue is used, softmax can change the chunksize.
    CHECK_EQ(chunksize, groups);
  }

  switch (type) {
  case PredictionType::kValue:
  case PredictionType::kMargin: {
    if (chunksize == 1 && !strict_shape) {
      *out_dim = 1;
      shape.resize(*out_dim);
      shape.front() = rows;
    } else {
      *out_dim = 2;
      shape.resize(*out_dim);
      shape.front() = rows;
      shape.back() = groups;
    }
    break;
  }
  case PredictionType::kApproxContribution:
  case PredictionType::kContribution: {
    auto groups = chunksize / (cols + 1);
    if (groups == 1 && !strict_shape) {
      *out_dim = 2;
      shape.resize(*out_dim);
      shape.front() = rows;
      shape.back() = cols + 1;
    } else {
      *out_dim = 3;
      shape.resize(*out_dim);
      shape[0] = rows;
      shape[1] = groups;
      shape[2] = cols + 1;
    }
    break;
  }
  case PredictionType::kInteraction: {
    if (groups == 1 && !strict_shape) {
      *out_dim = 3;
      shape.resize(*out_dim);
      shape[0] = rows;
      shape[1] = cols + 1;
      shape[2] = cols + 1;
    } else {
      *out_dim = 4;
      shape.resize(*out_dim);
      shape[0] = rows;
      shape[1] = groups;
      shape[2] = cols + 1;
      shape[3] = cols + 1;
    }
    break;
  }
  case PredictionType::kLeaf: {
    if (strict_shape) {
      shape.resize(4);
      shape[0] = rows;
      shape[1] = rounds;
      shape[2] = groups;
      auto forest = chunksize / (shape[1] * shape[2]);
      forest = std::max(static_cast<decltype(forest)>(1), forest);
      shape[3] = forest;
      *out_dim = shape.size();
    } else {
      *out_dim = 2;
      shape.resize(*out_dim);
      shape.front() = rows;
      shape.back() = chunksize;
    }
    break;
  }
  default: {
    LOG(FATAL) << "Unknown prediction type:" << static_cast<int>(type);
  }
  }
  CHECK_EQ(
      std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<>{}),
      chunksize * rows);
}
}  // namespace xgboost
#endif  // XGBOOST_C_API_C_API_UTILS_H_
