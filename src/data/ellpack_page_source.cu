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
  EllpackPage* page_;
  common::Monitor monitor_;
  dh::BulkAllocator ba_;
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
    return true;
  }

  bool Read(EllpackPage* page,
            dmlc::SeekStream* fi,
            const std::vector<bst_uint>& sorted_index_set) override {
    return true;
  }

  void Write(const EllpackPage& page, dmlc::Stream* fo) override {
    auto buffer = page.Impl()->idx_buffer;
    CHECK(!buffer.empty());
    fo->Write(buffer);
  }
};

EllpackPageSourceImpl::EllpackPageSourceImpl(
    DMatrix* dmat, const std::string& cache_info, const BatchParam& param) noexcept(false)
    : page_(nullptr) {
  monitor_.Init("ellpack_page_source");
  dh::safe_cuda(cudaSetDevice(param.gpu_id));

  monitor_.StartCuda("Quantiles");
  common::HistogramCuts hmat;
  size_t row_stride =
      common::DeviceSketch(param.gpu_id, param.max_bin, param.gpu_batch_nrows, dmat, &hmat);
  monitor_.StopCuda("Quantiles");

  monitor_.StartCuda("CreateInfo");
  EllpackInfo ellpack_info{param.gpu_id, dmat->IsDense(), row_stride, hmat, ba_};
  monitor_.StopCuda("CreateInfo");

  monitor_.StartCuda("WriteEllpackPages");
  const std::string page_type = ".ellpack.page";
  auto cinfo = ParseCacheInfo(cache_info, page_type);
  SparsePageWriter<EllpackPage> writer(cinfo.name_shards, cinfo.format_shards, 6);
  std::shared_ptr<EllpackPage> page;
  writer.Alloc(&page);
  auto* impl = page->Impl();
  impl->matrix.info = ellpack_info;
  impl->Clear();

  const MetaInfo& info = dmat->Info();
  size_t bytes_write = 0;
  double tstart = dmlc::GetTime();
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    impl->Push(param.gpu_id, batch);

    if (impl->MemCostBytes() >= DMatrix::kPageSize) {
      bytes_write += impl->MemCostBytes();
      writer.PushWrite(std::move(page));
      writer.Alloc(&page);
      impl = page->Impl();
      impl->matrix.info = ellpack_info;
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
}

void EllpackPageSourceImpl::BeforeFirst() {}

bool EllpackPageSourceImpl::Next() {
  return false;
}

EllpackPage& EllpackPageSourceImpl::Value() {
  return *page_;
}

const EllpackPage& EllpackPageSourceImpl::Value() const {
  return *page_;
}

XGBOOST_REGISTER_ELLPACK_PAGE_FORMAT(raw)
    .describe("Raw ELLPACK binary data format.")
    .set_body([]() {
      return new EllpackPageRawFormat();
    });

}  // namespace data
}  // namespace xgboost
