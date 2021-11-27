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
#include "../../../src/data/array_interface.h"
#include "../../../src/common/linalg_op.h"

namespace xgboost {
inline void TestMetaInfoStridedData(int32_t device) {
  MetaInfo info;
  {
    // label
    HostDeviceVector<float> labels;
    labels.Resize(64);
    auto& h_labels = labels.HostVector();
    std::iota(h_labels.begin(), h_labels.end(), 0.0f);
    bool is_gpu = device >= 0;
    if (is_gpu) {
      labels.SetDevice(0);
    }

    auto t = linalg::TensorView<float const, 2>{
        is_gpu ? labels.ConstDeviceSpan() : labels.ConstHostSpan(), {32, 2}, device};
    auto s = t.Slice(linalg::All(), 0);

    auto str = ArrayInterfaceStr(s);
    ASSERT_EQ(s.Size(), 32);

    info.SetInfo("label", StringView{str});
    auto const& h_result = info.labels_.HostVector();
    ASSERT_EQ(h_result.size(), 32);

    for (auto v : h_result) {
      ASSERT_EQ(static_cast<int32_t>(v) % 2, 0);
    }
  }
  {
    // qid
    linalg::Tensor<uint64_t, 2> qid;
    qid.Reshape(32, 2);
    auto& h_qid = qid.Data()->HostVector();
    std::iota(h_qid.begin(), h_qid.end(), 0);
    auto s = qid.View(device).Slice(linalg::All(), 0);
    auto str = ArrayInterfaceStr(s);
    info.SetInfo("qid", StringView{str});
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

    info.SetInfo("base_margin", StringView{ArrayInterfaceStr(t_margin)});
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
      return v_0;
    });
  }
}
}  // namespace xgboost
#endif  // XGBOOST_TESTS_CPP_DATA_TEST_METAINFO_H_
