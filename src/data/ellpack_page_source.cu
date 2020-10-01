/*!
 * Copyright 2019 XGBoost contributors
 */
#include <memory>
#include <utility>

#include "../common/hist_util.cuh"

#include "ellpack_page.cuh"
#include "ellpack_page_source.h"
#include "sparse_page_source.h"

namespace xgboost {
namespace data {

// Build the quantile sketch across the whole input data, then use the histogram cuts to compress
// each CSR page, and write the accumulated ELLPACK pages to disk.
EllpackPageSource::EllpackPageSource(DMatrix* dmat,
                                     const std::string& cache_info,
                                     const BatchParam& param) noexcept(false) {
  cache_info_ = ParseCacheInfo(cache_info, kPageType_);
  for (auto file : cache_info_.name_shards) {
    CheckCacheFileExists(file);
  }
  if (param.gpu_page_size > 0) {
    page_size_ = param.gpu_page_size;
  }

  monitor_.Init("ellpack_page_source");
  dh::safe_cuda(cudaSetDevice(param.gpu_id));

  monitor_.Start("Quantiles");
  size_t row_stride = GetRowStride(dmat);
  auto cuts = common::DeviceSketch(param.gpu_id, dmat, param.max_bin);
  monitor_.Stop("Quantiles");

  monitor_.Start("WriteEllpackPages");
  WriteEllpackPages(param.gpu_id, dmat, cuts, cache_info, row_stride);
  monitor_.Stop("WriteEllpackPages");

  external_prefetcher_.reset(
      new ExternalMemoryPrefetcher<EllpackPage>(cache_info_));
}

// Compress each CSR page to ELLPACK, and write the accumulated pages to disk.
void EllpackPageSource::WriteEllpackPages(int device, DMatrix* dmat,
                                          const common::HistogramCuts& cuts,
                                          const std::string& cache_info,
                                          size_t row_stride) const {
  auto cinfo = ParseCacheInfo(cache_info, kPageType_);
  const size_t extra_buffer_capacity = 6;
  SparsePageWriter<EllpackPage> writer(cinfo.name_shards, cinfo.format_shards,
                                       extra_buffer_capacity);
  std::shared_ptr<EllpackPage> page;
  SparsePage temp_host_page;
  writer.Alloc(&page);
  auto* impl = page->Impl();

  size_t bytes_write = 0;
  double tstart = dmlc::GetTime();
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    temp_host_page.Push(batch);

    size_t mem_cost_bytes =
        EllpackPageImpl::MemCostBytes(temp_host_page.Size(), row_stride, cuts);
    if (mem_cost_bytes >= page_size_) {
      bytes_write += mem_cost_bytes;
      *impl = EllpackPageImpl(device, cuts, temp_host_page, dmat->IsDense(),
                              row_stride);
      writer.PushWrite(std::move(page));
      writer.Alloc(&page);
      impl = page->Impl();
      temp_host_page.Clear();
      double tdiff = dmlc::GetTime() - tstart;
      LOG(INFO) << "Writing " << kPageType_ << " to " << cache_info << " in "
                << ((bytes_write >> 20UL) / tdiff) << " MB/s, "
                << (bytes_write >> 20UL) << " written";
    }
  }
  if (temp_host_page.Size() != 0) {
    *impl = EllpackPageImpl(device, cuts, temp_host_page, dmat->IsDense(),
                            row_stride);
    writer.PushWrite(std::move(page));
  }
}

}  // namespace data
}  // namespace xgboost
