/*!
 * Copyright 2019 XGBoost contributors
 */

#include "ellpack_page_source.h"

#include <memory>
#include <utility>

#include "../common/hist_util.h"

namespace xgboost {
namespace data {

EllpackPageSource::EllpackPageSource(DMatrix* dmat,
                                     const std::string& cache_info,
                                     const BatchParam& param) noexcept(false) : page_(dmat, param) {
  monitor_.Init("ellpack_page_source");
  dh::safe_cuda(cudaSetDevice(param.gpu_id));

  monitor_.StartCuda("Quantiles");
  common::HistogramCuts hmat;
  size_t row_stride = common::DeviceSketch(
      param.gpu_id, param.max_bin, param.gpu_batch_nrows, dmat, &hmat);
  monitor_.StopCuda("Quantiles");

  const std::string page_type = ".ellpack.page";
  auto cinfo = ParseCacheInfo(cache_info, page_type);
  SparsePageWriter<EllpackPage, EllpackPageFormat>
      writer(cinfo.name_shards, cinfo.format_shards, 6);
  std::shared_ptr<EllpackPage> page;
  writer.Alloc(&page);
  page->Clear();

  const MetaInfo& info = dmat->Info();
  size_t bytes_write = 0;
  double tstart = dmlc::GetTime();
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    page->Push(batch);

    if (page->MemCostBytes() >= DMatrix::kPageSize) {
      bytes_write += page->MemCostBytes();
      writer.PushWrite(std::move(page));
      writer.Alloc(&page);
      page->Clear();
      double tdiff = dmlc::GetTime() - tstart;
      LOG(INFO) << "Writing to " << cache_info << " in "
                << ((bytes_write >> 20UL) / tdiff) << " MB/s, "
                << (bytes_write >> 20UL) << " written";
    }
  }
  if (page->Size() != 0) {
    writer.PushWrite(std::move(page));
  }

  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(cinfo.name_info.c_str(), "w"));
  int tmagic = SparsePageSource<EllpackPage>::kMagic;
  fo->Write(&tmagic, sizeof(tmagic));
  info.SaveBinary(fo.get());
  LOG(INFO) << "EllpackPageSource: Finished writing to " << cinfo.name_info;
}

void EllpackPageSource::CreateEllpackPage(DMatrix* dmat, const std::string& cache_info) {}

XGBOOST_REGISTER_ELLPACK_PAGE_FORMAT(raw)
    .describe("Raw binary data format.")
    .set_body([]() {
      return new EllpackPageFormat();
    });

}  // namespace data
}  // namespace xgboost
