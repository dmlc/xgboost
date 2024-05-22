/**
 * Copyright 2020-2024 by XGBoost contributors
 */
#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "../../../src/data/gradient_index.h"       // for GHistIndexMatrix
#pragma GCC diagnostic pop

#include "../../../plugin/sycl/common/hist_util.h"
#include "../../../plugin/sycl/device_manager.h"
#include "sycl_helpers.h"
#include "../helpers.h"

namespace xgboost::sycl::common {

template <typename GradientSumT>
void GHistBuilderTest(float sparsity, bool force_atomic_use) {
  const size_t num_rows = 8;
  const size_t num_columns = 1;
  const int n_bins = 2;
  const GradientSumT eps = 1e-6;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  auto p_fmat = RandomDataGenerator{num_rows, num_columns, sparsity}.GenerateDMatrix();
  sycl::DeviceMatrix dmat;
  dmat.Init(qu, p_fmat.get());

  GHistIndexMatrix gmat_sycl;
  gmat_sycl.Init(qu, &ctx, dmat, n_bins);

  xgboost::GHistIndexMatrix gmat{&ctx, p_fmat.get(), n_bins, 0.3, false};

  RowSetCollection row_set_collection;
  auto& row_indices = row_set_collection.Data();
  row_indices.Resize(&qu, num_rows);
  size_t* p_row_indices = row_indices.Data();

  qu.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(num_rows),
                       [p_row_indices](::sycl::item<1> pid) {
      const size_t idx = pid.get_id(0);
      p_row_indices[idx] = idx;
    });
  }).wait_and_throw();
  row_set_collection.Init();

  auto builder = GHistBuilder<GradientSumT>(qu, n_bins);

  std::vector<GradientPair> gpair = {
      {0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}, {0.7f, 0.8f},
      {0.9f, 0.1f}, {0.2f, 0.3f}, {0.4f, 0.5f}, {0.6f, 0.7f}};
  CHECK_EQ(gpair.size(), num_rows);
  USMVector<GradientPair, MemoryType::on_device> gpair_device(&qu, gpair);

  std::vector<GradientSumT> hist_host(2*n_bins);
  GHistRow<GradientSumT, MemoryType::on_device> hist(&qu, 2 * n_bins);
  ::sycl::event event;

  const size_t nblocks = 2;
  GHistRow<GradientSumT, MemoryType::on_device> hist_buffer(&qu, 2 * nblocks * n_bins);

  InitHist(qu, &hist, hist.Size(), &event);
  InitHist(qu, &hist_buffer, hist_buffer.Size(), &event);

  event = builder.BuildHist(gpair_device, row_set_collection[0], gmat_sycl, &hist,
                            sparsity < eps , &hist_buffer, event, force_atomic_use);
  qu.memcpy(hist_host.data(), hist.Data(),
            2 * n_bins * sizeof(GradientSumT), event);
  qu.wait_and_throw();

  // Build hist on host to compare
  std::vector<GradientSumT> hist_desired(2*n_bins);
  for (size_t rid = 0; rid < num_rows; ++rid) {
    const size_t ibegin = gmat.row_ptr[rid];
    const size_t iend = gmat.row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      const size_t bin_idx = gmat.index[i];
      hist_desired[2*bin_idx]   += gpair[rid].GetGrad();
      hist_desired[2*bin_idx+1] += gpair[rid].GetHess();
    }
  }

  VerifySyclVector(hist_host, hist_desired, eps);
}

template <typename GradientSumT>
void GHistSubtractionTest() {
  const size_t n_bins = 4;
  using GHistType = GHistRow<GradientSumT, MemoryType::on_device>;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  ::sycl::event event;
  std::vector<GradientSumT> hist1_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  GHistType hist1(&qu, 2 * n_bins);
  event = qu.memcpy(hist1.Data(), hist1_host.data(),
                    2 * n_bins * sizeof(GradientSumT), event);

  std::vector<GradientSumT> hist2_host = {0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
  GHistType hist2(&qu, 2 * n_bins);
  event = qu.memcpy(hist2.Data(), hist2_host.data(),
            2 * n_bins * sizeof(GradientSumT), event);

  std::vector<GradientSumT> hist3_host(2 * n_bins);
  GHistType hist3(&qu, 2 * n_bins);
  event = SubtractionHist(qu, &hist3, hist1, hist2, n_bins, event);
  qu.memcpy(hist3_host.data(), hist3.Data(),
            2 * n_bins * sizeof(GradientSumT), event);
  qu.wait_and_throw();

  std::vector<GradientSumT> hist3_desired(2 * n_bins);
  for (size_t idx = 0; idx < 2 * n_bins; ++idx) {
    hist3_desired[idx] = hist1_host[idx] - hist2_host[idx];
  }

  const GradientSumT eps = 1e-6;
  VerifySyclVector(hist3_host, hist3_desired, eps);
}

TEST(SyclGHistBuilder, ByBlockDenseCase) {
  GHistBuilderTest<float>(0.0, false);
  GHistBuilderTest<double>(0.0, false);
}

TEST(SyclGHistBuilder, ByBlockSparseCase) {
  GHistBuilderTest<float>(0.3, false);
  GHistBuilderTest<double>(0.3, false);
}

TEST(SyclGHistBuilder, ByAtomicDenseCase) {
  GHistBuilderTest<float>(0.0, true);
  GHistBuilderTest<double>(0.0, true);
}

TEST(SyclGHistBuilder, ByAtomicSparseCase) {
  GHistBuilderTest<float>(0.3, true);
  GHistBuilderTest<double>(0.3, true);
}

TEST(SyclGHistBuilder, Subtraction) {
  GHistSubtractionTest<float>();
  GHistSubtractionTest<double>();
}

}  // namespace xgboost::sycl::common
