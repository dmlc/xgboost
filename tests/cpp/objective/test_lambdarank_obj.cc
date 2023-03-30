/**
 * Copyright 2023 by XGBoost Contributors
 */
#include "test_lambdarank_obj.h"

#include <gtest/gtest.h>                        // for Test, Message, TestPartResult, CmpHel...

#include <cstddef>                              // for size_t
#include <initializer_list>                     // for initializer_list
#include <map>                                  // for map
#include <memory>                               // for unique_ptr, shared_ptr, make_shared
#include <numeric>                              // for iota
#include <string>                               // for char_traits, basic_string, string
#include <vector>                               // for vector

#include "../../../src/common/ranking_utils.h"  // for LambdaRankParam
#include "../../../src/common/ranking_utils.h"  // for NDCGCache, LambdaRankParam
#include "../helpers.h"                         // for CheckRankingObjFunction, CheckConfigReload
#include "xgboost/base.h"                       // for GradientPair, bst_group_t, Args
#include "xgboost/context.h"                    // for Context
#include "xgboost/data.h"                       // for MetaInfo, DMatrix
#include "xgboost/host_device_vector.h"         // for HostDeviceVector
#include "xgboost/linalg.h"                     // for Tensor, All, TensorView
#include "xgboost/objective.h"                  // for ObjFunction
#include "xgboost/span.h"                       // for Span

namespace xgboost::obj {
void InitMakePairTest(Context const* ctx, MetaInfo* out_info, HostDeviceVector<float>* out_predt) {
  out_predt->SetDevice(ctx->gpu_id);
  MetaInfo& info = *out_info;
  info.num_row_ = 128;
  info.labels.ModifyInplace([&](HostDeviceVector<float>* data, common::Span<std::size_t> shape) {
    shape[0] = info.num_row_;
    shape[1] = 1;
    auto& h_data = data->HostVector();
    h_data.resize(shape[0]);
    for (std::size_t i = 0; i < h_data.size(); ++i) {
      h_data[i] = i % 2;
    }
  });
  std::vector<float> predt(info.num_row_);
  std::iota(predt.rbegin(), predt.rend(), 0.0f);
  out_predt->HostVector() = predt;
}

TEST(LambdaRank, MakePair) {
  Context ctx;
  MetaInfo info;
  HostDeviceVector<float> predt;

  InitMakePairTest(&ctx, &info, &predt);

  ltr::LambdaRankParam param;
  param.UpdateAllowUnknown(Args{{"lambdarank_pair_method", "topk"}});
  ASSERT_TRUE(param.HasTruncation());

  std::shared_ptr<ltr::RankingCache> p_cache = std::make_shared<ltr::NDCGCache>(&ctx, info, param);
  auto const& h_predt = predt.ConstHostVector();
  {
    auto rank_idx = p_cache->SortedIdx(&ctx, h_predt);
    for (std::size_t i = 0; i < h_predt.size(); ++i) {
      ASSERT_EQ(rank_idx[i], static_cast<std::size_t>(*(h_predt.crbegin() + i)));
    }
    std::int32_t n_pairs{0};
    MakePairs(&ctx, 0, p_cache, 0, info.labels.HostView().Slice(linalg::All(), 0), rank_idx,
              [&](auto i, auto j) {
                ASSERT_GT(j, i);
                ASSERT_LT(i, p_cache->Param().NumPair());
                ++n_pairs;
              });
    ASSERT_EQ(n_pairs, 3568);
  }

  auto const h_label = info.labels.HostView();

  {
    param.UpdateAllowUnknown(Args{{"lambdarank_pair_method", "mean"}});
    auto p_cache = std::make_shared<ltr::NDCGCache>(&ctx, info, param);
    ASSERT_FALSE(param.HasTruncation());
    std::int32_t n_pairs = 0;
    auto rank_idx = p_cache->SortedIdx(&ctx, h_predt);
    MakePairs(&ctx, 0, p_cache, 0, info.labels.HostView().Slice(linalg::All(), 0), rank_idx,
              [&](auto i, auto j) {
                ++n_pairs;
                // Not in the same bucket
                ASSERT_NE(h_label(rank_idx[i]), h_label(rank_idx[j]));
              });
    ASSERT_EQ(n_pairs, info.num_row_ * param.NumPair());
  }

  {
    param.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample", "2"}});
    auto p_cache = std::make_shared<ltr::NDCGCache>(&ctx, info, param);
    auto rank_idx = p_cache->SortedIdx(&ctx, h_predt);
    std::int32_t n_pairs = 0;
    MakePairs(&ctx, 0, p_cache, 0, info.labels.HostView().Slice(linalg::All(), 0), rank_idx,
              [&](auto i, auto j) {
                ++n_pairs;
                // Not in the same bucket
                ASSERT_NE(h_label(rank_idx[i]), h_label(rank_idx[j]));
              });
    ASSERT_EQ(param.NumPair(), 2);
    ASSERT_EQ(n_pairs, info.num_row_ * param.NumPair());
  }
}
}  // namespace xgboost::obj
