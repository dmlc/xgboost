/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#ifndef XGBOOST_TESTS_CPP_DATA_TEST_METAINFO_H_
#define XGBOOST_TESTS_CPP_DATA_TEST_METAINFO_H_
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/linalg.h>

#include <numeric>

#include "../../../src/common/linalg_op.h"

namespace xgboost {
inline void TestMetaInfoStridedData(DeviceOrd device) {
  MetaInfo info;
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", device.Name()}});
  {
    // labels
    linalg::Tensor<float, 3> labels;
    labels.Reshape(4, 2, 3);
    auto& h_label = labels.Data()->HostVector();
    std::iota(h_label.begin(), h_label.end(), 0.0);
    auto t_labels = labels.View(device).Slice(linalg::All(), 0, linalg::All());
    ASSERT_EQ(t_labels.Shape().size(), 2);

    info.SetInfo(ctx, "label", StringView{ArrayInterfaceStr(t_labels)});
    auto const& h_result = info.labels.View(DeviceOrd::CPU());
    ASSERT_EQ(h_result.Shape().size(), 2);
    auto in_labels = labels.View(DeviceOrd::CPU());
    linalg::ElementWiseKernelHost(h_result, omp_get_max_threads(), [&](size_t i, std::size_t j) {
      // Sliced at second dimension.
      auto v_0 = h_result(i, j);
      auto v_1 = in_labels(i, 0, j);
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
    auto const& h_result = info.base_margin_.View(DeviceOrd::CPU());
    ASSERT_EQ(h_result.Shape().size(), 2);
    auto in_margin = base_margin.View(DeviceOrd::CPU());
    linalg::ElementWiseKernelHost(h_result, omp_get_max_threads(),
                                  [&](std::size_t i, std::size_t j) {
                                    // Sliced at second dimension.
                                    auto v_0 = h_result(i, j);
                                    auto v_1 = in_margin(i, 0, j);
                                    CHECK_EQ(v_0, v_1);
                                  });
  }
}
}  // namespace xgboost
#endif  // XGBOOST_TESTS_CPP_DATA_TEST_METAINFO_H_
