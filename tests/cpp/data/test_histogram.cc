#include <gtest/gtest.h>
#include "../../../src/data/histogram.h"
#include "../../../src/common/hist_util.h"
#include "../helpers.h"
#include "../../../src/common/timer.h"

namespace xgboost {

void printCsr(std::vector<uint32_t> ptrs, std::vector<float> values) {
  for (auto p : ptrs) {
    std::cout << p << ", ";
  }
  std::cout << std::endl;
  for (size_t i = 1; i < ptrs.size(); ++i) {
    auto beg = ptrs[i-1];
    auto end = ptrs[i];
    for (size_t j = beg; j < end; ++j) {
      std::cout << values.at(j) << ", ";
    }
    std::cout << std::endl;
  }
}

TEST(CutMatrix, Build) {
  CutMatrix cuts;
  size_t constexpr kRows = 17;
  size_t constexpr kCols = 15;

  auto pp_mat = CreateDMatrix(kRows, kCols, 0);
  auto& p_mat = *pp_mat;
  common::Monitor m;
  m.Init("Test-Cut");

  common::HistCutMatrix hmat;
  m.Start("Old");
  hmat.Init(p_mat.get(), 256);
  m.Stop("Old");
  printCsr(hmat.row_ptr, hmat.cut);

  std::cout << std::endl;

  m.Start("New");
  cuts.Build(p_mat.get(), 256);
  m.Stop("New");
  printCsr(cuts.column_ptrs_, cuts.cut_values_);

  delete pp_mat;
}

}  // namespace xgboost
