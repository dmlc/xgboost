/*!
 * Copyright 2019 XGBoost contributors
 */

#include "ellpack_page_source.h"

#include <memory>
#include <utility>
#include <vector>

#include "../common/hist_util.h"
#include "ellpack_page.cuh"

namespace xgboost {
namespace data {

class EllpackPageSourceImpl : public DataSource<EllpackPage> {
 public:
  /*!
   * \brief Create source from cache files the cache_prefix.
   * \param cache_prefix The prefix of cache we want to solve.
   */
  explicit EllpackPageSourceImpl(DMatrix* dmat,
                                 const std::string& cache_info,
                                 const BatchParam& param) noexcept(false);

  /*! \brief destructor */
  ~EllpackPageSourceImpl() override = default;

  void BeforeFirst() override;
  bool Next() override;
  EllpackPage& Value();
  const EllpackPage& Value() const override;

 private:
  int device_{-1};
  common::Monitor monitor_;
  dh::BulkAllocator ba_;
  EllpackInfo ellpack_info_;
  std::unique_ptr<SparsePageSource<EllpackPage>> source_;
};

EllpackPageSource::EllpackPageSource(DMatrix* dmat,
                                     const std::string& cache_info,
                                     const BatchParam& param) noexcept(false)
    : impl_{new EllpackPageSourceImpl(dmat, cache_info, param)} {}

void EllpackPageSource::BeforeFirst() {
  impl_->BeforeFirst();
}

bool EllpackPageSource::Next() {
  return impl_->Next();
}

EllpackPage& EllpackPageSource::Value() {
  return impl_->Value();
}

const EllpackPage& EllpackPageSource::Value() const {
  return impl_->Value();
}

class EllpackPageRawFormat : public SparsePageFormat<EllpackPage> {
 public:
  bool Read(EllpackPage* page, dmlc::SeekStream* fi) override {
    return fi->Read(&page->Impl()->idx_buffer);
  }

  bool Read(EllpackPage* page,
            dmlc::SeekStream* fi,
            const std::vector<bst_uint>& sorted_index_set) override {
    return fi->Read(&page->Impl()->idx_buffer);
  }

  void Write(const EllpackPage& page, dmlc::Stream* fo) override {
    auto buffer = page.Impl()->idx_buffer;
    CHECK(!buffer.empty());
    fo->Write(buffer);
  }
};

EllpackPageSourceImpl::EllpackPageSourceImpl(DMatrix* dmat,
                                             const std::string& cache_info,
                                             const BatchParam& param) noexcept(false) {
  device_ = param.gpu_id;

  monitor_.Init("ellpack_page_source");
  dh::safe_cuda(cudaSetDevice(device_));

  monitor_.StartCuda("Quantiles");
  common::HistogramCuts hmat;
  size_t row_stride =
      common::DeviceSketch(device_, param.max_bin, param.gpu_batch_nrows, dmat, &hmat);
  monitor_.StopCuda("Quantiles");

  monitor_.StartCuda("CreateInfo");
  ellpack_info_ = EllpackInfo(device_, dmat->IsDense(), row_stride, hmat, ba_);
  monitor_.StopCuda("CreateInfo");

  monitor_.StartCuda("WriteEllpackPages");
  const std::string page_type = ".ellpack.page";
  auto cinfo = ParseCacheInfo(cache_info, page_type);
  SparsePageWriter<EllpackPage> writer(cinfo.name_shards, cinfo.format_shards, 6);
  std::shared_ptr<EllpackPage> page;
  writer.Alloc(&page);
  auto* impl = page->Impl();
  impl->matrix.info = ellpack_info_;
  impl->Clear();

  const MetaInfo& info = dmat->Info();
  size_t bytes_write = 0;
  double tstart = dmlc::GetTime();
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    impl->Push(device_, batch);

    if (impl->MemCostBytes() >= DMatrix::kPageSize) {
      bytes_write += impl->MemCostBytes();
      writer.PushWrite(std::move(page));
      writer.Alloc(&page);
      impl = page->Impl();
      impl->matrix.info = ellpack_info_;
      impl->Clear();
      double tdiff = dmlc::GetTime() - tstart;
      LOG(INFO) << "Writing to " << cache_info << " in "
                << ((bytes_write >> 20UL) / tdiff) << " MB/s, "
                << (bytes_write >> 20UL) << " written";
    }
  }
  if (impl->Size() != 0) {
    writer.PushWrite(std::move(page));
  }
  LOG(INFO) << "EllpackPageSource: Finished writing to " << cinfo.name_info;
  monitor_.StopCuda("WriteEllpackPages");

  source_.reset(new SparsePageSource<EllpackPage>(cache_info, page_type));
}

void EllpackPageSourceImpl::BeforeFirst() {
  source_->BeforeFirst();
}

bool EllpackPageSourceImpl::Next() {
  return source_->Next();
}

EllpackPage& EllpackPageSourceImpl::Value() {
  EllpackPage& page = source_->Value();
  page.Impl()->InitDevice(device_, ellpack_info_);
  return page;
}

const EllpackPage& EllpackPageSourceImpl::Value() const {
  EllpackPage& page = source_->Value();
  page.Impl()->InitDevice(device_, ellpack_info_);
  return page;
}

XGBOOST_REGISTER_ELLPACK_PAGE_FORMAT(raw)
    .describe("Raw ELLPACK binary data format.")
    .set_body([]() {
      return new EllpackPageRawFormat();
    });

}  // namespace data
}  // namespace xgboost
