/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "../common/categorical.h"    // for IsCat
#include "../common/column_matrix.h"  // for ColumnMatrix
#include "../common/common.h"         // for Range1d
#include "../common/hist_util.h"      // for DispatchBinType, HistogramCuts
#include "../common/math.h"           // for CheckNAN
#include "../data/cat_container.h"    // for NoOpAccessor
#include "../data/gradient_index.h"   // for GHistIndexMatrix
#include "xgboost/data.h"             // for HostSparsePageView
#include "xgboost/span.h"             // for Span
#include "xgboost/tree_model.h"       // for RegTree::FVec

namespace xgboost::predictor {
// Convert a single sample in batch view to FVec.
template <typename BatchView>
struct DataToFeatVec {
  void Fill(bst_idx_t ridx, RegTree::FVec* p_feats) const {
    auto& feats = *p_feats;
    auto n_valid = static_cast<BatchView const*>(this)->DoFill(ridx, feats.Data().data());
    feats.HasMissing(n_valid != feats.Size());
  }

  // Fill the data into the feature vector.
  void FVecFill(common::Range1d const& block, bst_feature_t n_features,
                common::Span<RegTree::FVec> s_feats_vec) const {
    auto feats_vec = s_feats_vec.data();
    for (std::size_t i = 0; i < block.Size(); ++i) {
      RegTree::FVec& feats = feats_vec[i];
      if (feats.Size() == 0) {
        feats.Init(n_features);
      }
      this->Fill(block.begin() + i, &feats);
    }
  }
  // Clear the feature vector.
  static void FVecDrop(common::Span<RegTree::FVec> s_feats) {
    auto p_feats = s_feats.data();
    for (size_t i = 0, n = s_feats.size(); i < n; ++i) {
      p_feats[i].Drop();
    }
  }
};

template <typename EncAccessor = NoOpAccessor>
class SparsePageView : public DataToFeatVec<SparsePageView<EncAccessor>> {
  EncAccessor acc_;
  HostSparsePageView const view_;

 public:
  bst_idx_t const base_rowid;

  SparsePageView(HostSparsePageView const p, bst_idx_t base_rowid, EncAccessor acc)
      : acc_{std::move(acc)}, view_{p}, base_rowid{base_rowid} {}

  [[nodiscard]] std::size_t Size() const { return view_.Size(); }

  [[nodiscard]] bst_idx_t DoFill(bst_idx_t ridx, float* out) const {
    auto p_data = view_[ridx].data();

    for (std::size_t i = 0, n = view_[ridx].size(); i < n; ++i) {
      auto const& entry = p_data[i];
      out[entry.index] = acc_(entry);
    }

    return view_[ridx].size();
  }
};

template <typename EncAccessor = NoOpAccessor>
class GHistIndexMatrixView : public DataToFeatVec<GHistIndexMatrixView<EncAccessor>> {
 private:
  GHistIndexMatrix const& page_;
  EncAccessor acc_;
  common::Span<FeatureType const> ft_;

  std::vector<std::uint32_t> const& ptrs_;
  std::vector<float> const& mins_;
  std::vector<float> const& values_;
  common::ColumnMatrix const& columns_;

 public:
  bst_idx_t const base_rowid;

 public:
  GHistIndexMatrixView(GHistIndexMatrix const& page, EncAccessor acc,
                       common::Span<FeatureType const> ft)
      : page_{page},
        acc_{std::move(acc)},
        ft_{ft},
        ptrs_{page.cut.Ptrs()},
        mins_{page.cut.MinValues()},
        values_{page.cut.Values()},
        columns_{page.Transpose()},
        base_rowid{page.base_rowid} {}

  [[nodiscard]] bst_idx_t DoFill(bst_idx_t ridx, float* out) const {
    auto gridx = ridx + this->base_rowid;
    auto n_features = page_.Features();

    bst_idx_t n_non_missings = 0;
    if (page_.IsDense()) {
      common::DispatchBinType(page_.index.GetBinTypeSize(), [&](auto t) {
        using T = decltype(t);
        auto ptr = this->page_.index.template data<T>();
        auto rbeg = this->page_.row_ptr[ridx];
        for (bst_feature_t fidx = 0; fidx < n_features; ++fidx) {
          bst_bin_t bin_idx;
          float fvalue;
          if (common::IsCat(ft_, fidx)) {
            bin_idx = page_.GetGindex(gridx, fidx);
            fvalue = this->values_[bin_idx];
          } else {
            bin_idx = ptr[rbeg + fidx] + page_.index.Offset()[fidx];
            fvalue =
                common::HistogramCuts::NumericBinValue(this->ptrs_, values_, mins_, fidx, bin_idx);
          }
          out[fidx] = acc_(fvalue, fidx);
        }
      });
      n_non_missings += n_features;
    } else {
      for (bst_feature_t fidx = 0; fidx < n_features; ++fidx) {
        float fvalue = std::numeric_limits<float>::quiet_NaN();
        bool is_cat = common::IsCat(ft_, fidx);
        if (columns_.GetColumnType(fidx) == common::kSparseColumn) {
          // Special handling for extremely sparse data. Just binary search.
          auto bin_idx = page_.GetGindex(gridx, fidx);
          if (bin_idx != -1) {
            if (is_cat) {
              fvalue = values_[bin_idx];
            } else {
              fvalue = common::HistogramCuts::NumericBinValue(this->ptrs_, values_, mins_, fidx,
                                                              bin_idx);
            }
          }
        } else {
          fvalue = page_.GetFvalue(ptrs_, values_, mins_, gridx, fidx, is_cat);
        }
        if (!common::CheckNAN(fvalue)) {
          out[fidx] = acc_(fvalue, fidx);
          n_non_missings++;
        }
      }
    }
    return n_non_missings;
  }

  [[nodiscard]] bst_idx_t Size() const { return page_.Size(); }
};

template <typename Adapter, typename EncAccessor = NoOpAccessor>
class AdapterView : public DataToFeatVec<AdapterView<Adapter, EncAccessor>> {
  Adapter const* adapter_;
  float missing_;
  EncAccessor acc_;

 public:
  explicit AdapterView(Adapter const* adapter, float missing, EncAccessor acc)
      : adapter_{adapter}, missing_{missing}, acc_{std::move(acc)} {}

  [[nodiscard]] bst_idx_t DoFill(bst_idx_t ridx, float* out) const {
    auto const& batch = adapter_->Value();
    auto row = batch.GetLine(ridx);
    bst_idx_t n_non_missings = 0;
    for (size_t c = 0; c < row.Size(); ++c) {
      auto e = row.GetElement(c);
      if (missing_ != e.value && !common::CheckNAN(e.value)) {
        auto fvalue = this->acc_(e);
        out[e.column_idx] = fvalue;
        n_non_missings++;
      }
    }
    return n_non_missings;
  }

  [[nodiscard]] bst_idx_t Size() const { return adapter_->NumRows(); }

  bst_idx_t const static base_rowid = 0;  // NOLINT
};
}  // namespace xgboost::predictor
