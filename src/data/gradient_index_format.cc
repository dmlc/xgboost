/**
 * Copyright 2021-2023 XGBoost contributors
 */
#include <cstddef>      // for size_t
#include <cstdint>      // for uint8_t
#include <type_traits>  // for underlying_type_t
#include <vector>       // for vector

#include "../common/io.h"                 // for AlignedResourceReadStream
#include "../common/ref_resource_view.h"  // for ReadVec, WriteVec
#include "gradient_index.h"               // for GHistIndexMatrix
#include "histogram_cut_format.h"         // for ReadHistogramCuts
#include "sparse_page_writer.h"           // for SparsePageFormat

namespace xgboost::data {
class GHistIndexRawFormat : public SparsePageFormat<GHistIndexMatrix> {
 public:
  bool Read(GHistIndexMatrix* page, common::AlignedResourceReadStream* fi) override {
    CHECK(fi);

    if (!ReadHistogramCuts(&page->cut, fi)) {
      return false;
    }

    // indptr
    if (!common::ReadVec(fi, &page->row_ptr)) {
      return false;
    }

    // data
    // - bin type
    // Old gcc doesn't support reading from enum.
    std::underlying_type_t<common::BinTypeSize> uint_bin_type{0};
    if (!fi->Read(&uint_bin_type)) {
      return false;
    }
    common::BinTypeSize size_type = static_cast<common::BinTypeSize>(uint_bin_type);
    // - index buffer
    if (!common::ReadVec(fi, &page->data)) {
      return false;
    }
    // - index
    page->index = common::Index{common::Span{page->data.data(), page->data.size()}, size_type};

    // hit count
    if (!common::ReadVec(fi, &page->hit_count)) {
      return false;
    }
    if (!fi->Read(&page->max_numeric_bins_per_feat)) {
      return false;
    }
    if (!fi->Read(&page->base_rowid)) {
      return false;
    }
    bool is_dense = false;
    if (!fi->Read(&is_dense)) {
      return false;
    }
    page->SetDense(is_dense);
    if (is_dense) {
      page->index.SetBinOffset(page->cut.Ptrs());
    }

    if (!page->ReadColumnPage(fi)) {
      return false;
    }
    return true;
  }

  std::size_t Write(GHistIndexMatrix const& page, common::AlignedFileWriteStream* fo) override {
    std::size_t bytes = 0;
    bytes += WriteHistogramCuts(page.cut, fo);
    // indptr
    bytes += common::WriteVec(fo, page.row_ptr);

    // data
    // - bin type
    std::underlying_type_t<common::BinTypeSize> uint_bin_type = page.index.GetBinTypeSize();
    bytes += fo->Write(uint_bin_type);
    // - index buffer
    std::vector<std::uint8_t> data(page.index.begin(), page.index.end());
    bytes += fo->Write(static_cast<std::uint64_t>(data.size()));
    if (!data.empty()) {
      bytes += fo->Write(data.data(), data.size());
    }

    // hit count
    bytes += common::WriteVec(fo, page.hit_count);
    // max_bins, base row, is_dense
    bytes += fo->Write(page.max_numeric_bins_per_feat);
    bytes += fo->Write(page.base_rowid);
    bytes += fo->Write(page.IsDense());

    bytes += page.WriteColumnPage(fo);
    return bytes;
  }
};

DMLC_REGISTRY_FILE_TAG(gradient_index_format);

XGBOOST_REGISTER_GHIST_INDEX_PAGE_FORMAT(raw)
    .describe("Raw GHistIndex binary data format.")
    .set_body([]() { return new GHistIndexRawFormat(); });
}  // namespace xgboost::data
