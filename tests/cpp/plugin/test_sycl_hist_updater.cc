/**
 * Copyright 2020-2024 by XGBoost contributors
 */
#include <gtest/gtest.h>

#include <oneapi/dpl/random>

#include "../../../plugin/sycl/tree/hist_updater.h"
#include "../../../plugin/sycl/device_manager.h"

#include "../helpers.h"

namespace xgboost::sycl::tree {

template <typename GradientSumT>
class TestHistUpdater : public HistUpdater<GradientSumT> {
 public:
  TestHistUpdater(::sycl::queue qu,
                  const xgboost::tree::TrainParam& param,
                  std::unique_ptr<TreeUpdater> pruner,
                  FeatureInteractionConstraintHost int_constraints_,
                  DMatrix const* fmat) : HistUpdater<GradientSumT>(qu, param, std::move(pruner),
                                                                   int_constraints_, fmat) {}

  void TestInitSampling(const USMVector<GradientPair, MemoryType::on_device> &gpair,
                    USMVector<size_t, MemoryType::on_device>* row_indices) {
    HistUpdater<GradientSumT>::InitSampling(gpair, row_indices);
  }
};

template <typename GradientSumT>
void TestHistUpdaterSampling(const xgboost::tree::TrainParam& param) {
  const size_t num_rows = 1u << 12;
  const size_t num_columns = 1;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());
  ObjInfo task{ObjInfo::kRegression};

  auto p_fmat = RandomDataGenerator{num_rows, num_columns, 0.0}.GenerateDMatrix();

  FeatureInteractionConstraintHost int_constraints;
  std::unique_ptr<TreeUpdater> pruner{TreeUpdater::Create("prune", &ctx, &task)};

  TestHistUpdater<GradientSumT> updater(qu, param, std::move(pruner), int_constraints, p_fmat.get());

  USMVector<size_t, MemoryType::on_device> row_indices_0(&qu, num_rows);
  USMVector<size_t, MemoryType::on_device> row_indices_1(&qu, num_rows);
  USMVector<GradientPair, MemoryType::on_device> gpair(&qu, num_rows);
  auto* gpair_ptr = gpair.Data();
  qu.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(::sycl::range<1>(num_rows)),
                                        [=](::sycl::item<1> pid) {
      uint64_t i = pid.get_linear_id();

      constexpr uint32_t seed = 777;
      oneapi::dpl::minstd_rand engine(seed, i);
      oneapi::dpl::uniform_real_distribution<GradientPair::ValueT> distr(-1., 1.);
      gpair_ptr[i] = {distr(engine), distr(engine)};
    });
  }).wait();

  updater.TestInitSampling(gpair, &row_indices_0);
  
  size_t n_samples = row_indices_0.Size();
  // Half of gpairs have neg hess
  ASSERT_LT(n_samples, num_rows * 0.5 * param.subsample * 1.2);
  ASSERT_GT(n_samples, num_rows * 0.5 * param.subsample / 1.2);

  // Check if two lanunches generate different realisations:
  updater.TestInitSampling(gpair, &row_indices_1);
  if (row_indices_1.Size() == n_samples) {
    std::vector<size_t> row_indices_0_host(n_samples);
    std::vector<size_t> row_indices_1_host(n_samples);
    qu.memcpy(row_indices_0_host.data(), row_indices_0.Data(), n_samples * sizeof(size_t)).wait();
    qu.memcpy(row_indices_1_host.data(), row_indices_1.Data(), n_samples * sizeof(size_t)).wait();

    // The order in row_indices_0 and row_indices_1 can be different
    std::set<size_t> rows;
    for (auto row : row_indices_0_host) {
      rows.insert(row);
    }

    size_t num_diffs = 0;
    for (auto row : row_indices_1_host) {
      if (rows.count(row) == 0) num_diffs++;
    }

    ASSERT_NE(num_diffs, 0);
  }

}

TEST(SyclHistUpdater, Sampling) {
  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"subsample", "0.7"}});

  TestHistUpdaterSampling<float>(param);
  TestHistUpdaterSampling<double>(param);
}
}  // namespace xgboost::sycl::tree
