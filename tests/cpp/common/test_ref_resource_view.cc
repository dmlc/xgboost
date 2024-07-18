/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstddef>  // for size_t
#include <memory>   // for make_shared, make_unique
#include <numeric>  // for iota
#include <vector>   // for vector

#include "../../../src/common/ref_resource_view.h"
#include "dmlc/filesystem.h"  // for TemporaryDirectory

namespace xgboost::common {
TEST(RefResourceView, Basic) {
  std::size_t n_bytes = 1024;
  auto mem = std::make_shared<MallocResource>(n_bytes);
  {
    RefResourceView view{static_cast<float*>(mem->Data()), mem->Size() / sizeof(float), mem};

    RefResourceView kview{static_cast<float const*>(mem->Data()), mem->Size() / sizeof(float), mem};
    ASSERT_EQ(mem.use_count(), 3);
    ASSERT_EQ(view.size(), n_bytes / sizeof(1024));
    ASSERT_EQ(kview.size(), n_bytes / sizeof(1024));
  }
  {
    RefResourceView view{static_cast<float*>(mem->Data()), mem->Size() / sizeof(float), mem};
    std::fill_n(static_cast<float*>(mem->Data()), mem->Size() / sizeof(float), 1.5f);
    for (auto v : view) {
      ASSERT_EQ(v, 1.5f);
    }
    std::iota(view.begin(), view.end(), 0.0f);
    ASSERT_EQ(view.front(), 0.0f);
    ASSERT_EQ(view.back(), static_cast<float>(view.size() - 1));

    view.front() = 1.0f;
    view.back() = 2.0f;
    ASSERT_EQ(view.front(), 1.0f);
    ASSERT_EQ(view.back(), 2.0f);
  }
  ASSERT_EQ(mem.use_count(), 1);
}

TEST(RefResourceView, IO) {
  dmlc::TemporaryDirectory tmpdir;
  auto path = tmpdir.path + "/testfile";
  auto data = MakeFixedVecWithMalloc(123, std::size_t{1});

  {
    auto fo = std::make_unique<AlignedFileWriteStream>(StringView{path}, "wb");
    ASSERT_EQ(fo->Write(data.data(), data.size_bytes()), data.size_bytes());
  }
  {
    auto fo = std::make_unique<AlignedFileWriteStream>(StringView{path}, "wb");
    ASSERT_EQ(WriteVec(fo.get(), data),
              data.size_bytes() + sizeof(RefResourceView<std::size_t>::size_type));
  }
  {
    auto fi = std::make_unique<PrivateMmapConstStream>(
        path, 0, data.size_bytes() + sizeof(RefResourceView<std::size_t>::size_type));
    auto read = MakeFixedVecWithMalloc(123, std::size_t{1});
    ASSERT_TRUE(ReadVec(fi.get(), &read));
    for (auto v : read) {
      ASSERT_EQ(v, 1ul);
    }
  }
}

TEST(RefResourceView, IOAligned) {
  dmlc::TemporaryDirectory tmpdir;
  auto path = tmpdir.path + "/testfile";
  auto data = MakeFixedVecWithMalloc(123, 1.0f);

  {
    auto fo = std::make_unique<AlignedFileWriteStream>(StringView{path}, "wb");
    // + sizeof(float) for alignment
    ASSERT_EQ(WriteVec(fo.get(), data),
              data.size_bytes() + sizeof(RefResourceView<std::size_t>::size_type) + sizeof(float));
  }
  {
    auto fi = std::make_unique<PrivateMmapConstStream>(
        path, 0, data.size_bytes() + sizeof(RefResourceView<std::size_t>::size_type));
    // wrong type, float vs. double
    auto read = MakeFixedVecWithMalloc(123, 2.0);
    ASSERT_FALSE(ReadVec(fi.get(), &read));
  }
  {
    auto fi = std::make_unique<PrivateMmapConstStream>(
        path, 0, data.size_bytes() + sizeof(RefResourceView<std::size_t>::size_type));
    auto read = MakeFixedVecWithMalloc(123, 2.0f);
    ASSERT_TRUE(ReadVec(fi.get(), &read));
    for (auto v : read) {
      ASSERT_EQ(v, 1ul);
    }
  }
  {
    // Test std::vector
    std::vector<float> data(123);
    std::iota(data.begin(), data.end(), 0.0f);
    auto fo = std::make_unique<AlignedFileWriteStream>(StringView{path}, "wb");
    // + sizeof(float) for alignment
    ASSERT_EQ(WriteVec(fo.get(), data), data.size() * sizeof(float) +
                                            sizeof(RefResourceView<std::size_t>::size_type) +
                                            sizeof(float));
  }
}
}  // namespace xgboost::common
