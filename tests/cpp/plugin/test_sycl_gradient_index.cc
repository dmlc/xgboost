/**
 * Copyright 2021-2024 by XGBoost contributors
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "../../../src/data/gradient_index.h"       // for GHistIndexMatrix
#pragma GCC diagnostic pop

#include "../../../plugin/sycl/data/gradient_index.h"
#include "../../../plugin/sycl/device_manager.h"
#include "sycl_helpers.h"
#include "../helpers.h"

namespace xgboost::sycl::data {

TEST(SyclGradientIndex, HistogramCuts) {
  size_t max_bins = 8;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  auto p_fmat = RandomDataGenerator{512, 16, 0.5}.GenerateDMatrix(true);

  xgboost::common::HistogramCuts cut = 
    xgboost::common::SketchOnDMatrix(&ctx, p_fmat.get(), max_bins);

  common::HistogramCuts cut_sycl;
  cut_sycl.Init(qu, cut);

  VerifySyclVector(cut_sycl.Ptrs(), cut.cut_ptrs_.HostVector());
  VerifySyclVector(cut_sycl.Values(), cut.cut_values_.HostVector());
  VerifySyclVector(cut_sycl.MinValues(), cut.min_vals_.HostVector());
}

TEST(SyclGradientIndex, Init) {
  size_t n_rows = 128;
  size_t n_columns = 7;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  auto p_fmat = RandomDataGenerator{n_rows, n_columns, 0.3}.GenerateDMatrix();

  sycl::DeviceMatrix dmat;
  dmat.Init(qu, p_fmat.get());

  int max_bins = 256;
  common::GHistIndexMatrix gmat_sycl;
  gmat_sycl.Init(qu, &ctx, dmat, max_bins);

  xgboost::GHistIndexMatrix gmat{&ctx, p_fmat.get(), max_bins, 0.3, false};

  {
    ASSERT_EQ(gmat_sycl.max_num_bins, max_bins);
    ASSERT_EQ(gmat_sycl.nfeatures, n_columns);
  }

  {
    VerifySyclVector(gmat_sycl.hit_count, gmat.hit_count);
  }

  {
    std::vector<size_t> feature_count_sycl(n_columns, 0);
    gmat_sycl.GetFeatureCounts(feature_count_sycl.data());

    std::vector<size_t> feature_count(n_columns, 0);
    gmat.GetFeatureCounts(feature_count.data());
    VerifySyclVector(feature_count_sycl, feature_count);
  }
}

}  // namespace xgboost::sycl::data
