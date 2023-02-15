/*!
 * Copyright 2021-2022 XGBoost contributors
 */
#include "sparse_page_writer.h"
#include "gradient_index.h"
#include "histogram_cut_format.h"

namespace xgboost {
namespace data {
class GHistIndexRawFormat : public SparsePageFormat<GHistIndexMatrix> {
 public:
  bool Read(GHistIndexMatrix* page, dmlc::SeekStream* fi) override {
    if (!ReadHistogramCuts(&page->cut, fi)) {
      return false;
    }
    // indptr
    fi->Read(&page->row_ptr);
    // data
    std::vector<uint8_t> data;
    if (!fi->Read(&data)) {
      return false;
    }
    page->index.Resize(data.size());
    std::copy(data.cbegin(), data.cend(), page->index.begin());
    // bin type
    // Old gcc doesn't support reading from enum.
    std::underlying_type_t<common::BinTypeSize> uint_bin_type{0};
    if (!fi->Read(&uint_bin_type)) {
      return false;
    }
    common::BinTypeSize size_type =
        static_cast<common::BinTypeSize>(uint_bin_type);
    page->index.SetBinTypeSize(size_type);
    // hit count
    if (!fi->Read(&page->hit_count)) {
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

    page->ReadColumnPage(fi);
    return true;
  }

  size_t Write(GHistIndexMatrix const &page, dmlc::Stream *fo) override {
    size_t bytes = 0;
    bytes += WriteHistogramCuts(page.cut, fo);
    // indptr
    fo->Write(page.row_ptr);
    bytes += page.row_ptr.size() * sizeof(decltype(page.row_ptr)::value_type) +
             sizeof(uint64_t);
    // data
    std::vector<uint8_t> data(page.index.begin(), page.index.end());
    fo->Write(data);
    bytes += data.size() * sizeof(decltype(data)::value_type) + sizeof(uint64_t);
    // bin type
    std::underlying_type_t<common::BinTypeSize> uint_bin_type =
        page.index.GetBinTypeSize();
    fo->Write(uint_bin_type);
    bytes += sizeof(page.index.GetBinTypeSize());
    // hit count
    fo->Write(page.hit_count);
    bytes +=
        page.hit_count.size() * sizeof(decltype(page.hit_count)::value_type) +
        sizeof(uint64_t);
    // max_bins, base row, is_dense
    fo->Write(page.max_numeric_bins_per_feat);
    bytes += sizeof(page.max_numeric_bins_per_feat);
    fo->Write(page.base_rowid);
    bytes += sizeof(page.base_rowid);
    fo->Write(page.IsDense());
    bytes += sizeof(page.IsDense());

    bytes += page.WriteColumnPage(fo);
    return bytes;
  }
};

DMLC_REGISTRY_FILE_TAG(gradient_index_format);

XGBOOST_REGISTER_GHIST_INDEX_PAGE_FORMAT(raw)
    .describe("Raw GHistIndex binary data format.")
    .set_body([]() { return new GHistIndexRawFormat(); });

}  // namespace data
}  // namespace xgboost
