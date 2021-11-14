/*!
 * Copyright 2021 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/linalg.h>

#include <numeric>

#include "../../../src/common/linalg_op.h"

namespace xgboost {
namespace linalg {
namespace {
auto kCpuId = GenericParameter::kCpuId;
}

auto MakeMatrixFromTest(HostDeviceVector<float> *storage, size_t n_rows, size_t n_cols) {
  storage->Resize(n_rows * n_cols);
  auto &h_storage = storage->HostVector();

  std::iota(h_storage.begin(), h_storage.end(), 0);

  auto m = linalg::TensorView<float, 2>{h_storage, {n_rows, static_cast<size_t>(n_cols)}, -1};
  return m;
}

TEST(Linalg, MatrixView) {
  size_t kRows = 31, kCols = 77;
  HostDeviceVector<float> storage;
  auto m = MakeMatrixFromTest(&storage, kRows, kCols);
  ASSERT_EQ(m.DeviceIdx(), kCpuId);
  ASSERT_EQ(m(0, 0), 0);
  ASSERT_EQ(m(kRows - 1, kCols - 1), storage.Size() - 1);
}

TEST(Linalg, VectorView) {
  size_t kRows = 31, kCols = 77;
  HostDeviceVector<float> storage;
  auto m = MakeMatrixFromTest(&storage, kRows, kCols);
  auto v = m.Slice(linalg::All(), 3);
  for (size_t i = 0; i < v.Size(); ++i) {
    ASSERT_EQ(v(i), m(i, 3));
  }

  ASSERT_EQ(v(0), 3);
}

TEST(Linalg, TensorView) {
  std::vector<double> data(2 * 3 * 4, 0);
  std::iota(data.begin(), data.end(), 0);

  TensorView<double> t{data, {2, 3, 4}, -1};
  ASSERT_EQ(t.Shape()[0], 2);
  ASSERT_EQ(t.Shape()[1], 3);
  ASSERT_EQ(t.Shape()[2], 4);

  float v = t(0, 1, 2);
  ASSERT_EQ(v, 6);

  auto s = t.Slice(1, All(), All());
  ASSERT_EQ(s.Shape().size(), 2);
  ASSERT_EQ(s.Shape()[0], 3);
  ASSERT_EQ(s.Shape()[1], 4);

  std::vector<std::vector<double>> sol{
      {12.0, 13.0, 14.0, 15.0}, {16.0, 17.0, 18.0, 19.0}, {20.0, 21.0, 22.0, 23.0}};
  for (size_t i = 0; i < s.Shape()[0]; ++i) {
    for (size_t j = 0; j < s.Shape()[1]; ++j) {
      ASSERT_EQ(s(i, j), sol[i][j]);
    }
  }

  {
    // as vector
    TensorView<double, 1> vec{data, {data.size()}, -1};
    ASSERT_EQ(vec.Size(), data.size());
    ASSERT_EQ(vec.Shape(0), data.size());
    ASSERT_EQ(vec.Shape().size(), 1);
    for (size_t i = 0; i < data.size(); ++i) {
      ASSERT_EQ(vec(i), data[i]);
    }
  }

  {
    // as matrix
    TensorView<double, 2> mat(data, {6, 4}, -1);
    auto s = mat.Slice(2, All());
    ASSERT_EQ(s.Shape().size(), 1);
    s = mat.Slice(All(), 1);
    ASSERT_EQ(s.Shape().size(), 1);
  }

  {
    // assignment
    TensorView<double, 3> t{data, {2, 3, 4}, 0};
    double pi = 3.14159;
    t(1, 2, 3) = pi;
    ASSERT_EQ(t(1, 2, 3), pi);
  }

  {
    // Don't assign the initial dimension, tensor should be able to deduce the correct dim
    // for Slice.
    TensorView<double> t{data, {2, 3, 4}, 0};
    auto s = t.Slice(1, 2, All());
    static_assert(decltype(s)::kDimension == 1, "");
  }
}

TEST(Linalg, Tensor) {
  {
    Tensor<float, 3> t{{2, 3, 4}, kCpuId};
    auto view = t.View(kCpuId);

    auto const &as_const = t;
    auto k_view = as_const.View(kCpuId);

    size_t n = 2 * 3 * 4;
    ASSERT_EQ(t.Size(), n);
    ASSERT_TRUE(std::equal(k_view.cbegin(), k_view.cbegin(), view.begin()));

    Tensor<float, 3> t_0{std::move(t)};
    ASSERT_EQ(t_0.Size(), n);
    ASSERT_EQ(t_0.Shape(0), 2);
    ASSERT_EQ(t_0.Shape(1), 3);
    ASSERT_EQ(t_0.Shape(2), 4);
  }
  {
    // Reshape
    Tensor<float, 3> t{{2, 3, 4}, kCpuId};
    t.Reshape(4, 3, 2);
    ASSERT_EQ(t.Size(), 24);
    ASSERT_EQ(t.Shape(2), 2);
    t.Reshape(1);
    ASSERT_EQ(t.Size(), 1);
    t.Reshape(0, 0, 0);
    ASSERT_EQ(t.Size(), 0);
    t.Reshape(0, 3, 0);
    ASSERT_EQ(t.Size(), 0);
    ASSERT_EQ(t.Shape(1), 3);
    t.Reshape(3, 3, 3);
    ASSERT_EQ(t.Size(), 27);
  }
}

TEST(Linalg, Empty) {
  {
    auto t = TensorView<double, 2>{{}, {0, 3}, kCpuId};
    for (int32_t i : {0, 1, 2}) {
      auto s = t.Slice(All(), i);
      ASSERT_EQ(s.Size(), 0);
      ASSERT_EQ(s.Shape().size(), 1);
      ASSERT_EQ(s.Shape(0), 0);
    }
  }
  {
    auto t = Tensor<double, 2>{{0, 3}, kCpuId};
    ASSERT_EQ(t.Size(), 0);
    auto view = t.View(kCpuId);

    for (int32_t i : {0, 1, 2}) {
      auto s = view.Slice(All(), i);
      ASSERT_EQ(s.Size(), 0);
      ASSERT_EQ(s.Shape().size(), 1);
      ASSERT_EQ(s.Shape(0), 0);
    }
  }
}

TEST(Linalg, ArrayInterface) {
  auto cpu = kCpuId;
  auto t = Tensor<double, 2>{{3, 3}, cpu};
  auto v = t.View(cpu);
  std::iota(v.begin(), v.end(), 0);
  auto arr = Json::Load(StringView{v.ArrayInterfaceStr()});
  ASSERT_EQ(get<Integer>(arr["shape"][0]), 3);
  ASSERT_EQ(get<Integer>(arr["strides"][0]), 3 * sizeof(double));

  ASSERT_FALSE(get<Boolean>(arr["data"][1]));
  ASSERT_EQ(reinterpret_cast<double *>(get<Integer>(arr["data"][0])), v.Values().data());
}

TEST(Linalg, Popc) {
  {
    uint32_t v{0};
    ASSERT_EQ(detail::NativePopc(v), 0);
    ASSERT_EQ(detail::Popc(v), 0);
    v = 1;
    ASSERT_EQ(detail::NativePopc(v), 1);
    ASSERT_EQ(detail::Popc(v), 1);
    v = 0xffffffff;
    ASSERT_EQ(detail::NativePopc(v), 32);
    ASSERT_EQ(detail::Popc(v), 32);
  }
  {
    uint64_t v{0};
    ASSERT_EQ(detail::NativePopc(v), 0);
    ASSERT_EQ(detail::Popc(v), 0);
    v = 1;
    ASSERT_EQ(detail::NativePopc(v), 1);
    ASSERT_EQ(detail::Popc(v), 1);
    v = 0xffffffff;
    ASSERT_EQ(detail::NativePopc(v), 32);
    ASSERT_EQ(detail::Popc(v), 32);
    v = 0xffffffffffffffff;
    ASSERT_EQ(detail::NativePopc(v), 64);
    ASSERT_EQ(detail::Popc(v), 64);
  }
}

TEST(Linalg, Stack) {
  Tensor<float, 3> l{{2, 3, 4}, kCpuId};
  ElementWiseKernelHost(l.View(kCpuId), omp_get_max_threads(),
                        [=](size_t i, float v) { return i; });
  Tensor<float, 3> r_0{{2, 3, 4}, kCpuId};
  ElementWiseKernelHost(r_0.View(kCpuId), omp_get_max_threads(),
                        [=](size_t i, float v) { return i; });

  Stack(&l, r_0);

  Tensor<float, 3> r_1{{0, 3, 4}, kCpuId};
  Stack(&l, r_1);
  ASSERT_EQ(l.Shape(0), 4);

  Stack(&r_1, l);
  ASSERT_EQ(r_1.Shape(0), l.Shape(0));
}
}  // namespace linalg
}  // namespace xgboost
