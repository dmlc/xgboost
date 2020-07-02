/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <vector>
#include <xgboost/span.h>

#include "../helpers.h"
#include "../../../src/tree/updater_exact.h"

namespace xgboost {
namespace tree {

class MultiExactTest : public :: testing::Test {
 protected:
  static bst_row_t constexpr kRows { 64 };
  static bst_feature_t constexpr kCols { 16 };
  static bst_feature_t constexpr kLabels{16};

  HostDeviceVector<GradientPair> gradients_;
  std::shared_ptr<DMatrix> p_dmat_ {nullptr};

  void SetUp() override {
    gradients_ = GenerateRandomGradients(kRows * kLabels, -1.0f, 1.0f);
    auto h_grad = common::Span<GradientPair>{gradients_.HostVector()};
    p_dmat_ = RandomDataGenerator(kRows, kCols, .5f).GenerateDMatrix(true);
    p_dmat_->Info().labels_.Resize(kRows);

    auto &h_labels = p_dmat_->Info().labels_.HostVector();
    h_labels.resize(kRows * kLabels);
    SimpleLCG gen;
    xgboost::SimpleRealUniformDistribution<bst_float> dist(0, 1);

    for (auto &v : h_labels) {
      v = dist(&gen);
    }
    p_dmat_->Info().labels_cols = kCols;
    p_dmat_->Info().labels_rows = kRows;
  }

  ~MultiExactTest() override = default;
};


class MultiExactUpdaterForTest : public MultiExact<MultiGradientPair> {
 public:
  explicit MultiExactUpdaterForTest(GenericParameter const *runtime)
      : MultiExact{runtime} {
    this->Configure(Args{});
  }
  decltype(gpairs_) &GetGpairs() { return gpairs_; }
  decltype(positions_) &GetPositions() { return positions_; }
  decltype(nodes_split_) & GetNodesSplit() { return nodes_split_; }
};


TEST_F(MultiExactTest, InitData) {
  GenericParameter runtime;
  runtime.InitAllowUnknown(Args{});
  runtime.gpu_id = GenericParameter::kCpuId;
  MultiExactUpdaterForTest updater(&runtime);
  auto h_grad = common::Span<GradientPair>{gradients_.HostVector()};
  updater.InitData(p_dmat_.get(), h_grad, kLabels);

  auto const& gpairs = updater.GetGpairs();

  ASSERT_EQ(gpairs.size(), p_dmat_->Info().num_row_);
  for (size_t i = 0; i < gpairs.size(); ++i) {
    auto const& pair = gpairs[i];
    auto const& grad = pair.GetGrad();
    auto const& hess = pair.GetHess();
    ASSERT_EQ(pair.GetGrad().Size(), p_dmat_->Info().labels_cols);

    for (size_t j = 0; j < grad.Size(); ++j) {
      ASSERT_EQ(grad[j], h_grad[i * p_dmat_->Info().labels_cols + j].GetGrad());
      ASSERT_EQ(hess[j], h_grad[i * p_dmat_->Info().labels_cols + j].GetHess());
    }
  }

  ASSERT_TRUE(updater.GetPositions().empty());
  ASSERT_TRUE(updater.GetNodesSplit().empty());
}

TEST_F(MultiExactTest, InitRoot) {
  RegTree tree(p_dmat_->Info().num_col_, RegTree::kMulti);
  GenericParameter runtime;
  runtime.InitAllowUnknown(Args{});
  runtime.gpu_id = GenericParameter::kCpuId;
  MultiExactUpdaterForTest updater{&runtime};
  updater.Configure(Args{});
  auto h_grad = common::Span<GradientPair>{gradients_.HostVector()};
  updater.InitData(p_dmat_.get(), h_grad, kLabels);

  updater.InitRoot(p_dmat_.get(), &tree);
  auto root_weight = tree.VectorLeafValue(RegTree::kRoot);
  ASSERT_EQ(root_weight.size(), p_dmat_->Info().labels_cols);
  ASSERT_EQ(updater.GetPositions().size(), p_dmat_->Info().num_row_);
  ASSERT_EQ(updater.GetNodesSplit().front().nidx, RegTree::kRoot);
}

TEST_F(MultiExactTest, EvaluateSplit) {
  RegTree tree(p_dmat_->Info().num_col_, RegTree::kMulti);
  GenericParameter runtime;
  runtime.InitAllowUnknown(Args{});
  runtime.gpu_id = GenericParameter::kCpuId;
  MultiExactUpdaterForTest updater{&runtime};

  for (auto& page : p_dmat_->GetBatches<SortedCSCPage>()) {
    auto& offset = page.offset.HostVector();
    auto& data = page.data.HostVector();
    // No need for forward search for 0^th column.
    data[offset[0]] = data[offset[1] - 1];
  }

  updater.Configure(Args{});
  auto h_grad = common::Span<GradientPair>{gradients_.HostVector()};
  updater.InitData(p_dmat_.get(), h_grad, kLabels);
  updater.InitRoot(p_dmat_.get(), &tree);
  updater.GetNodesSplit().front().candidate.loss_chg = 0.001;

  std::vector<bst_feature_t> features { 0 };
  updater.EvaluateSplit(p_dmat_.get(), features);

  ASSERT_FALSE(updater.GetNodesSplit().front().candidate.DefaultLeft());
  ASSERT_EQ(updater.GetNodesSplit().front().candidate.SplitIndex(), 0);
}

TEST_F(MultiExactTest, ApplySplit) {
  RegTree tree(p_dmat_->Info().num_col_, RegTree::kMulti);
  GenericParameter runtime;
  runtime.InitAllowUnknown(Args{});
  runtime.gpu_id = GenericParameter::kCpuId;
  MultiExactUpdaterForTest updater{&runtime};
  updater.Configure(Args{});
  auto h_grad = common::Span<GradientPair>{gradients_.HostVector()};
  updater.InitData(p_dmat_.get(), h_grad, kLabels);
  updater.InitRoot(p_dmat_.get(), &tree);
  ASSERT_EQ(updater.GetNodesSplit().size(), 1);

  // Invent a valid split entry.
  auto left_sum =
      MakeGradientPair<MultiGradientPair>(p_dmat_->Info().labels_cols);
  auto right_sum =
      MakeGradientPair<MultiGradientPair>(p_dmat_->Info().labels_cols);
  for (size_t i = 0; i < p_dmat_->Info().labels_cols; ++i) {
    left_sum.GetGrad()[i] = 1.3f;
    left_sum.GetHess()[i] = 1.0f;
  }
  float split_value = 0.6;
  bst_feature_t split_ind = 0;
  updater.GetNodesSplit().front().candidate.loss_chg = 1.0f;
  auto success = updater.GetNodesSplit().front().candidate.Update(
      2.0f, split_ind, split_value, true, left_sum, right_sum);
  ASSERT_TRUE(success);
  ASSERT_TRUE(updater.GetNodesSplit().front().candidate.DefaultLeft());
  updater.ApplySplit(p_dmat_.get(), &tree);
  ASSERT_EQ(tree.NumExtraNodes(), 2);

  auto const& pos = updater.GetPositions();
  ASSERT_EQ(pos.size(), p_dmat_->Info().num_row_);
  std::set<bst_feature_t> non_missing;
  for (auto const& page : p_dmat_->GetBatches<SortedCSCPage>()) {
    auto column = page[split_ind];
    for (auto const& e : column) {
      if (e.fvalue < split_value) {
        ASSERT_EQ(pos[e.index], 1);
      } else {
        ASSERT_EQ(pos[e.index], 2);
      }
      non_missing.insert(e.index);
    }

    for (bst_row_t i = 0; i < pos.size(); ++i) {
      if (non_missing.find(i) == non_missing.cend()) {
        // dft left
        ASSERT_EQ(pos[i], 1);
      }
    }
  }
}
}  // namespace tree
}  // namespace xgboost
