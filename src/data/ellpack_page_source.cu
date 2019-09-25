/*!
 * Copyright 2019 XGBoost contributors
 */

#include "ellpack_page_source.h"

#include <memory>
#include <utility>

namespace xgboost {
namespace data {

EllpackPageSource::EllpackPageSource(DMatrix* src,
                                     const std::string& cache_info,
                                     const BatchParam& param) noexcept(false) : page_(src, param) {
  const std::string page_type = ".ellpack.page";
  auto cinfo = ParseCacheInfo(cache_info, page_type);
  {
    SparsePageWriter writer(cinfo.name_shards, cinfo.format_shards, 6);
    std::shared_ptr<SparsePage> page;
    writer.Alloc(&page);
    page->Clear();

    MetaInfo info = src->Info();
    size_t bytes_write = 0;
    double tstart = dmlc::GetTime();
    for (auto& batch : src->GetBatches<SparsePage>()) {
      if (page_type == ".col.page") {
        page->PushCSC(batch.GetTranspose(src->Info().num_col_));
      } else if (page_type == ".sorted.col.page") {
        SparsePage tmp = batch.GetTranspose(src->Info().num_col_);
        page->PushCSC(tmp);
        page->SortRows();
      } else {
        LOG(FATAL) << "Unknown page type: " << page_type;
      }

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
    if (page->data.Size() != 0) {
      writer.PushWrite(std::move(page));
    }

    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(cinfo.name_info.c_str(), "w"));
    int tmagic = SparsePageSource<EllpackPage>::kMagic;
    fo->Write(&tmagic, sizeof(tmagic));
    info.SaveBinary(fo.get());
  }
  LOG(INFO) << "SparsePageSource: Finished writing to " << cinfo.name_info;
}

void EllpackPageSource::CreateEllpackPage(DMatrix* dmat, const std::string& cache_info) {}

}  // namespace data
}  // namespace xgboost
