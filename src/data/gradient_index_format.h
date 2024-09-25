/**
 * Copyright 2021-2024, XGBoost contributors
 */
#pragma once

#include <cstddef>  // for size_t
#include <utility>  // for move

#include "../common/hist_util.h"  // for HistogramCuts
#include "../common/io.h"         // for AlignedFileWriteStream
#include "gradient_index.h"       // for GHistIndexMatrix
#include "sparse_page_writer.h"   // for SparsePageFormat

namespace xgboost::common {
class HistogramCuts;
}

namespace xgboost::data {
class GHistIndexRawFormat : public SparsePageFormat<GHistIndexMatrix> {
  common::HistogramCuts cuts_;

 public:
  [[nodiscard]] bool Read(GHistIndexMatrix* page, common::AlignedResourceReadStream* fi) override;
  [[nodiscard]] std::size_t Write(GHistIndexMatrix const& page,
                                  common::AlignedFileWriteStream* fo) override;

  explicit GHistIndexRawFormat(common::HistogramCuts cuts) : cuts_{std::move(cuts)} {}
};
}  // namespace xgboost::data
