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
  }
};

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
  SparsePageWriter<EllpackPage> writer(cinfo.name_shards, cinfo.format_shards, 6);
  std::shared_ptr<EllpackPage> page;
  writer.Alloc(&page);
  page->Impl()->InitInfo(param.gpu_id, row_stride, dmat->IsDense(), hmat);
  page->Clear();

  const MetaInfo& info = dmat->Info();
  size_t bytes_write = 0;
  double tstart = dmlc::GetTime();
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    page->Impl()->Push(row_stride, hmat, batch);

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
  int tmagic = SparsePageSource<SparsePage>::kMagic;
  fo->Write(&tmagic, sizeof(tmagic));
  info.SaveBinary(fo.get());
  LOG(INFO) << "EllpackPageSource: Finished writing to " << cinfo.name_info;
}

XGBOOST_REGISTER_ELLPACK_PAGE_FORMAT(raw)
    .describe("Raw ELLPACK binary data format.")
    .set_body([]() {
      return new EllpackPageRawFormat();
    });

}  // namespace data
}  // namespace xgboost
