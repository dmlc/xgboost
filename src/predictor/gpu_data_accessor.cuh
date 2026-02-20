/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#pragma once

#include <limits>
#include <type_traits>
#include <utility>

#include "../common/categorical.h"  // for IsCat
#include "xgboost/context.h"        // for Context
#include "xgboost/data.h"           // for Entry, SparsePage
#include "xgboost/span.h"           // for Span

namespace xgboost::predictor {
struct SparsePageView {
  common::Span<const Entry> d_data;
  common::Span<const bst_idx_t> d_row_ptr;
  bst_feature_t num_features;

  SparsePageView() = default;
  explicit SparsePageView(Context const* ctx, SparsePage const& page, bst_feature_t n_features)
      : d_data{[&] {
          page.data.SetDevice(ctx->Device());
          return page.data.ConstDeviceSpan();
        }()},
        d_row_ptr{[&] {
          page.offset.SetDevice(ctx->Device());
          return page.offset.ConstDeviceSpan();
        }()},
        num_features{n_features} {}

  [[nodiscard]] __device__ float GetElement(size_t ridx, size_t fidx) const {
    // Binary search
    auto begin_ptr = d_data.begin() + d_row_ptr[ridx];
    auto end_ptr = d_data.begin() + d_row_ptr[ridx + 1];
    if (end_ptr - begin_ptr == this->NumCols()) {
      // Bypass span check for dense data
      return d_data.data()[d_row_ptr[ridx] + fidx].fvalue;
    }
    common::Span<const Entry>::iterator previous_middle;
    while (end_ptr != begin_ptr) {
      auto middle = begin_ptr + (end_ptr - begin_ptr) / 2;
      if (middle == previous_middle) {
        break;
      } else {
        previous_middle = middle;
      }

      if (middle->index == fidx) {
        return middle->fvalue;
      } else if (middle->index < fidx) {
        begin_ptr = middle;
      } else {
        end_ptr = middle;
      }
    }
    // Value is missing
    return std::numeric_limits<float>::quiet_NaN();
  }

  [[nodiscard]] XGBOOST_DEVICE size_t NumRows() const { return d_row_ptr.size() - 1; }
  [[nodiscard]] XGBOOST_DEVICE size_t NumCols() const { return num_features; }
};

template <typename EncAccessor>
struct SparsePageLoaderNoShared {
 public:
  using SupportShmemLoad = std::false_type;

  SparsePageView data;
  EncAccessor acc;

  template <typename Fidx>
  [[nodiscard]] __device__ float GetElement(bst_idx_t ridx, Fidx fidx) const {
    return acc(data.GetElement(ridx, fidx), fidx);
  }
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumRows() const { return data.NumRows(); }
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumCols() const { return data.NumCols(); }
};

template <typename Accessor, typename EncAccessor>
struct EllpackLoader {
 public:
  using SupportShmemLoad = std::false_type;

  Accessor matrix;
  EncAccessor acc;

  XGBOOST_DEVICE EllpackLoader(Accessor m, bool /*use_shared*/, bst_feature_t /*n_features*/,
                               bst_idx_t /*n_samples*/, float /*missing*/, EncAccessor&& acc)
      : matrix{std::move(m)}, acc{std::forward<EncAccessor>(acc)} {}

  [[nodiscard]] XGBOOST_DEV_INLINE float GetElement(size_t ridx, size_t fidx) const {
    auto gidx = matrix.template GetBinIndex<false>(ridx, fidx);
    if (gidx == -1) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    if (common::IsCat(matrix.feature_types, fidx)) {
      return this->acc(matrix.gidx_fvalue_map[gidx], fidx);
    }
    // The gradient index needs to be shifted by one as min values are not included in the
    // cuts.
    if (gidx == matrix.feature_segments[fidx]) {
      return matrix.min_fvalue[fidx];
    }
    return matrix.gidx_fvalue_map[gidx - 1];
  }

  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumCols() const { return this->matrix.NumFeatures(); }
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumRows() const { return this->matrix.n_rows; }
};
}  // namespace xgboost::predictor
