/*!
 * Copyright 2019 XGBoost contributors
 */
#include <memory>
#include <utility>
#include <vector>

#include "../common/hist_util.h"

#include "ellpack_page_source.h"
#include "sparse_page_source.h"
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
  /*! \brief Write Ellpack pages after accumulating them in memory. */
  void WriteEllpackPages(DMatrix* dmat, const std::string& cache_info) const;

  /*! \brief The page type string for ELLPACK. */
  const std::string kPageType_{".ellpack.page"};

  int device_{-1};
  size_t page_size_{DMatrix::kPageSize};
  common::Monitor monitor_;
  dh::BulkAllocator ba_;
  /*! \brief The EllpackInfo, with the underlying GPU memory shared by all pages. */
  EllpackInfo ellpack_info_;
  std::unique_ptr<SparsePageSource<EllpackPage>> source_;
  std::string cache_info_;
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

// Build the quantile sketch across the whole input data, then use the histogram cuts to compress
// each CSR page, and write the accumulated ELLPACK pages to disk.
EllpackPageSourceImpl::EllpackPageSourceImpl(DMatrix* dmat,
                                             const std::string& cache_info,
                                             const BatchParam& param) noexcept(false)
    : device_(param.gpu_id), cache_info_(cache_info) {

  if (param.gpu_page_size > 0) {
    page_size_ = param.gpu_page_size;
  }

  monitor_.Init("ellpack_page_source");
  dh::safe_cuda(cudaSetDevice(device_));

  monitor_.StartCuda("Quantiles");
  common::HistogramCuts hmat;
  size_t row_stride =
      common::DeviceSketch(device_, param.max_bin, param.gpu_batch_nrows, dmat, &hmat);
  monitor_.StopCuda("Quantiles");

  monitor_.StartCuda("CreateEllpackInfo");
  ellpack_info_ = EllpackInfo(device_, dmat->IsDense(), row_stride, hmat, &ba_);
  monitor_.StopCuda("CreateEllpackInfo");

  monitor_.StartCuda("WriteEllpackPages");
  WriteEllpackPages(dmat, cache_info);
  monitor_.StopCuda("WriteEllpackPages");

  source_.reset(new SparsePageSource<EllpackPage>(cache_info_, kPageType_));
}

void EllpackPageSourceImpl::BeforeFirst() {
  source_.reset(new SparsePageSource<EllpackPage>(cache_info_, kPageType_));
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

// Compress each CSR page to ELLPACK, and write the accumulated pages to disk.
void EllpackPageSourceImpl::WriteEllpackPages(DMatrix* dmat, const std::string& cache_info) const {
  auto cinfo = ParseCacheInfo(cache_info, kPageType_);
  const size_t extra_buffer_capacity = 6;
  SparsePageWriter<EllpackPage> writer(
      cinfo.name_shards, cinfo.format_shards, extra_buffer_capacity);
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

    size_t mem_cost_bytes = impl->MemCostBytes();
    if (mem_cost_bytes >= page_size_) {
      bytes_write += mem_cost_bytes;
      impl->CompressSparsePage(device_);
      writer.PushWrite(std::move(page));
      writer.Alloc(&page);
      impl = page->Impl();
      impl->matrix.info = ellpack_info_;
      impl->Clear();
      double tdiff = dmlc::GetTime() - tstart;
      LOG(INFO) << "Writing " << kPageType_ << " to " << cache_info << " in "
                << ((bytes_write >> 20UL) / tdiff) << " MB/s, "
                << (bytes_write >> 20UL) << " written";
    }
  }
  if (impl->Size() != 0) {
    impl->CompressSparsePage(device_);
    writer.PushWrite(std::move(page));
  }
}

}  // namespace data
}  // namespace xgboost
