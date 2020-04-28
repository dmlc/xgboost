#include <gtest/gtest.h>

#include "../helpers.h"
#include "../../../src/data/iterative_device_dmatrix.h"
#include "../../../src/data/ellpack_page.cuh"
#include "../../../src/data/device_adapter.cuh"
#include "../../../src/data/device_dmatrix.h"

namespace xgboost {
namespace data {

class CudaArrayIterForTest {
  HostDeviceVector<float> data_;
  size_t iter_ {0};
  DMatrixHandle proxy_;
  std::unique_ptr<RandomDataGenerator> rng_;

  std::vector<std::string> batches_;
  std::string interface_;

 public:
  size_t static constexpr kRows { 1000 };
  size_t static constexpr kBatches { 100 };
  size_t static constexpr kCols { 13 };

  CudaArrayIterForTest() {
    XGProxyDMatrixCreate(&proxy_);
    rng_.reset(new RandomDataGenerator{kRows, kCols, 0.5});
    rng_->Device(0);
    std::tie(batches_, interface_) =
        rng_->GenerateArrayInterfaceBatch(&data_, kBatches);
    this->Reset();
  }
  ~CudaArrayIterForTest() {
    XGDMatrixFree(proxy_);
  }

  int Next() {
    if (iter_ == kBatches) {
      return 0;
    }
    XGDMatrixSetDataCudaArrayInterface(proxy_, batches_[iter_].c_str());
    iter_++;
    return 1;
  }
  std::string AsArray() const {
    return interface_;
  }
  void Reset() {
    iter_ = 0;
  }
  size_t Iter() const { return iter_; }
  auto Proxy() -> decltype(proxy_) { return proxy_; }
};

void Reset(DataIterHandle self) {
  static_cast<CudaArrayIterForTest*>(self)->Reset();
}

int Next(DataIterHandle self) {
  return static_cast<CudaArrayIterForTest*>(self)->Next();
}

TEST(IterativeDeviceDMatrix, Basic) {
  CudaArrayIterForTest iter;
  IterativeDeviceDMatrix m(&iter, iter.Proxy(), Reset, Next,
                           std::numeric_limits<float>::quiet_NaN(),
                           0, 256);
  size_t offset = 0;
  auto first = (*m.GetEllpackBatches({}).begin()).Impl();
  std::unique_ptr<EllpackPageImpl> page_concatenated {
    new EllpackPageImpl(0, first->Cuts(), first->is_dense,
                        first->row_stride, 1000 * 100)};
  for (auto& batch : m.GetBatches<EllpackPage>()) {
    auto page = batch.Impl();
    size_t num_elements = page_concatenated->Copy(0, page, offset);
    offset += num_elements;
  }
  auto from_iter = page_concatenated->GetDeviceAccessor(0);

  std::string interface = iter.AsArray();
  auto adapter = CupyAdapter(interface);
  auto dm = DeviceDMatrix(&adapter, std::numeric_limits<float>::quiet_NaN(), 0, 256);
  auto dm_impl = (*dm.GetEllpackBatches({}).begin()).Impl();
  auto from_data = dm_impl->GetDeviceAccessor(0);

  std::vector<float> cuts_from_iter(from_iter.gidx_fvalue_map.size());
  std::vector<float> min_fvalues_iter(from_iter.min_fvalue.size());
  std::vector<uint32_t> cut_ptrs_iter(from_iter.feature_segments.size());
  dh::CopyDeviceSpanToVector(&cuts_from_iter, from_iter.gidx_fvalue_map);
  dh::CopyDeviceSpanToVector(&min_fvalues_iter, from_iter.min_fvalue);
  dh::CopyDeviceSpanToVector(&cut_ptrs_iter, from_iter.feature_segments);

  std::vector<float> cuts_from_data(from_data.gidx_fvalue_map.size());
  std::vector<float> min_fvalues_data(from_data.min_fvalue.size());
  std::vector<uint32_t> cut_ptrs_data(from_data.feature_segments.size());
  dh::CopyDeviceSpanToVector(&cuts_from_data, from_data.gidx_fvalue_map);
  dh::CopyDeviceSpanToVector(&min_fvalues_data, from_data.min_fvalue);
  dh::CopyDeviceSpanToVector(&cut_ptrs_data, from_data.feature_segments);

  ASSERT_EQ(cuts_from_iter.size(), cuts_from_data.size());
  for (size_t i = 0; i < cuts_from_iter.size(); ++i) {
    EXPECT_NEAR(cuts_from_iter[i], cuts_from_data[i], kRtEps);
  }
  ASSERT_EQ(min_fvalues_iter.size(), min_fvalues_data.size());
  for (size_t i = 0; i < min_fvalues_iter.size(); ++i) {
    ASSERT_NEAR(min_fvalues_iter[i], min_fvalues_data[i], kRtEps);
  }
  ASSERT_EQ(cut_ptrs_iter.size(), cut_ptrs_data.size());
  for (size_t i = 0; i < cut_ptrs_iter.size(); ++i) {
    ASSERT_EQ(cut_ptrs_iter[i], cut_ptrs_data[i]);
  }
}
}  // namespace data
}  // namespace xgboost
