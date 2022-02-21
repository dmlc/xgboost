/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_TESTS_CPP_DATA_TEST_METAINFO_H_
#define XGBOOST_TESTS_CPP_DATA_TEST_METAINFO_H_
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/linalg.h>

#include <numeric>

#include "../../../src/common/linalg_op.h"
#include "../../../src/data/array_interface.h"

namespace xgboost {
inline void TestMetaInfoStridedData(int32_t device) {
  MetaInfo info;
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"gpu_id", std::to_string(device)}});
  {
    // labels
    linalg::Tensor<float, 3> labels;
    labels.Reshape(4, 2, 3);
    auto& h_label = labels.Data()->HostVector();
    std::iota(h_label.begin(), h_label.end(), 0.0);
    auto t_labels = labels.View(device).Slice(linalg::All(), 0, linalg::All());
    ASSERT_EQ(t_labels.Shape().size(), 2);

    info.SetInfo(ctx, "label", StringView{ArrayInterfaceStr(t_labels)});
    auto const& h_result = info.labels.View(-1);
    ASSERT_EQ(h_result.Shape().size(), 2);
    auto in_labels = labels.View(-1);
    linalg::ElementWiseKernelHost(h_result, omp_get_max_threads(), [&](size_t i, float& v_0) {
      auto tup = linalg::UnravelIndex(i, h_result.Shape());
      auto i0 = std::get<0>(tup);
      auto i1 = std::get<1>(tup);
      // Sliced at second dimension.
      auto v_1 = in_labels(i0, 0, i1);
      CHECK_EQ(v_0, v_1);
    });
  }
  {
    // qid
    linalg::Tensor<uint64_t, 2> qid;
    qid.Reshape(32, 2);
    auto& h_qid = qid.Data()->HostVector();
    std::iota(h_qid.begin(), h_qid.end(), 0);
    auto s = qid.View(device).Slice(linalg::All(), 0);
    auto str = ArrayInterfaceStr(s);
    info.SetInfo(ctx, "qid", StringView{str});
    auto const& h_result = info.group_ptr_;
    ASSERT_EQ(h_result.size(), s.Size() + 1);
  }
  {
    // base margin
    linalg::Tensor<float, 3> base_margin;
    base_margin.Reshape(4, 2, 3);
    auto& h_margin = base_margin.Data()->HostVector();
    std::iota(h_margin.begin(), h_margin.end(), 0.0);
    auto t_margin = base_margin.View(device).Slice(linalg::All(), 0, linalg::All());
    ASSERT_EQ(t_margin.Shape().size(), 2);

    info.SetInfo(ctx, "base_margin", StringView{ArrayInterfaceStr(t_margin)});
    auto const& h_result = info.base_margin_.View(-1);
    ASSERT_EQ(h_result.Shape().size(), 2);
    auto in_margin = base_margin.View(-1);
    linalg::ElementWiseKernelHost(h_result, omp_get_max_threads(), [&](size_t i, float v_0) {
      auto tup = linalg::UnravelIndex(i, h_result.Shape());
      auto i0 = std::get<0>(tup);
      auto i1 = std::get<1>(tup);
      // Sliced at second dimension.
      auto v_1 = in_margin(i0, 0, i1);
      CHECK_EQ(v_0, v_1);
    });
  }
}
}  // namespace xgboost
#endif  // XGBOOST_TESTS_CPP_DATA_TEST_METAINFO_H_
