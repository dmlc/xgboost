/*!
 * Copyright (c) 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_C_API_C_API_UTILS_H_
#define XGBOOST_C_API_C_API_UTILS_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "xgboost/logging.h"
#include "xgboost/json.h"
#include "xgboost/learner.h"
#include "xgboost/c_api.h"

#include "c_api_error.h"

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
  case PredictionType::kApproxInteraction:
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
    } else if (chunksize == 1) {
      *out_dim = 1;
      shape.resize(*out_dim);
      shape.front() = rows;
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

// Reverse the ntree_limit in old prediction API.
inline uint32_t GetIterationFromTreeLimit(uint32_t ntree_limit, Learner *learner) {
  // On Python and R, `best_ntree_limit` is set to `best_iteration * num_parallel_tree`.
  // To reverse it we just divide it by `num_parallel_tree`.
  if (ntree_limit != 0) {
    learner->Configure();
    uint32_t num_parallel_tree = 0;

    Json config{Object()};
    learner->SaveConfig(&config);
    auto const &booster =
        get<String const>(config["learner"]["gradient_booster"]["name"]);
    if (booster == "gblinear") {
      num_parallel_tree = 0;
    } else if (booster == "dart") {
      num_parallel_tree = std::stoi(
          get<String const>(config["learner"]["gradient_booster"]["gbtree"]
                                  ["gbtree_train_param"]["num_parallel_tree"]));
    } else if (booster == "gbtree") {
      num_parallel_tree = std::stoi(get<String const>(
          (config["learner"]["gradient_booster"]["gbtree_train_param"]
                 ["num_parallel_tree"])));
    } else {
      LOG(FATAL) << "Unknown booster:" << booster;
    }
    ntree_limit /= std::max(num_parallel_tree, 1u);
  }
  return ntree_limit;
}

inline float GetMissing(Json const &config) {
  float missing;
  auto const& j_missing = config["missing"];
  if (IsA<Number const>(j_missing)) {
    missing = get<Number const>(j_missing);
  } else if (IsA<Integer const>(j_missing)) {
    missing = get<Integer const>(j_missing);
  } else {
    missing = nan("");
    LOG(FATAL) << "Invalid missing value: " << j_missing;
  }
  return missing;
}
}  // namespace xgboost
#endif  // XGBOOST_C_API_C_API_UTILS_H_
