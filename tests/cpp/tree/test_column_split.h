/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once

#include <xgboost/data.h>          // for FeatureType, DMatrix

#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr
#include <vector>   // for vector

#include "../helpers.h"                 // for RandomDataGenerator

namespace xgboost::tree {
inline std::shared_ptr<DMatrix> GenerateCatDMatrix(std::size_t rows, std::size_t cols,
                                                   float sparsity, bool categorical) {
  if (categorical) {
    std::vector<FeatureType> ft(cols);
    for (size_t i = 0; i < ft.size(); ++i) {
      ft[i] = (i % 3 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
    }
    return RandomDataGenerator(rows, cols, sparsity)
        .Seed(3)
        .Type(ft)
        .MaxCategory(17)
        .GenerateDMatrix();
  } else {
    return RandomDataGenerator{rows, cols, sparsity}.Seed(3).GenerateDMatrix();
  }
}

void TestColumnSplit(bst_target_t n_targets, bool categorical, std::string name, float sparsity);
}  // namespace xgboost::tree
