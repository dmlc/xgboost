#include <xgboost/logging.h>
#include "histogram.h"

namespace xgboost {

CutMatrix::CutMatrix() : column_ptrs_{0} {}

void CutMatrix::Build(DMatrix* dmat, uint32_t const max_num_bins) {
  using WXQSketch = common::WXQuantileSketch<bst_float, bst_float>;
  auto& info = dmat->Info();
  constexpr float kFactor = 8;

  for (auto const& batch : dmat->GetColumnBatches()) {
    for (uint32_t col_id = 0; col_id < batch.Size(); ++col_id) {
      WXQSketch sketch;
      common::Span<xgboost::Entry const> column = batch[col_id];
      uint32_t n_bins = std::min(static_cast<uint32_t>(column.size()), max_num_bins);
      sketch.Init(info.num_row_, 1.0 / (n_bins * kFactor));
      for (auto const& entry : column) {
        auto row_idx = entry.index;
        sketch.Push(entry.fvalue, info.GetWeight(entry.index));
      }

      WXQSketch::SummaryContainer out_summary;
      sketch.GetSummary(&out_summary);
      WXQSketch::SummaryContainer summary;
      summary.Reserve(n_bins * kFactor);
      summary.SetPrune(out_summary, n_bins * kFactor);
      for (size_t i = 2; i < summary.size; ++i) {
        bst_float cut_point = summary.data[i-1].value;
        if (i == 2 || cut_point > cut_values_.back() ) {
          cut_values_.emplace_back(cut_point);
        }
      }

      bst_float cpt = summary.data[summary.size - 1].value;
      cpt += fabs(cpt) + 1e-5;
      cut_values_.emplace_back(cpt);

      auto cut_size = cut_values_.size();
      column_ptrs_.emplace_back(cut_size);
    }
  }
}

BinIdx CutMatrix::SearchBin(float value, uint32_t column_id) const {
  auto beg = column_ptrs_.at(column_id);
  auto end = column_ptrs_.at(column_id + 1);
  auto it = std::upper_bound(cut_values_.cbegin() + beg, cut_values_.cend() + end, value);
  if (it == cut_values_.cend()) {
    it = cut_values_.cend() - 1;
  }
  BinIdx idx = it - cut_values_.cbegin();
  return idx;
}

HistogramIndices::HistogramIndices() : column_ptrs_{0} {}

void HistogramIndices::Build(
    DMatrix* p_fmat, CutMatrix const& cut, uint32_t const max_num_bins) {
  for (auto const& batch : p_fmat->GetColumnBatches()) {
    for (auto col_id = 0; col_id < batch.Size(); ++col_id) {
      common::Span<xgboost::Entry const> column = batch[col_id];
      for (auto entry : column) {
        auto idx = cut.SearchBin(entry.fvalue, col_id);
        indices_.emplace_back(idx);
      }
      column_ptrs_.emplace_back(indices_.size());
    }
  }
}

}  // namespace xgboost
