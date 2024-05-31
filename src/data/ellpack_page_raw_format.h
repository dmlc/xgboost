/**
 * Copyright 2019-2024, XGBoost contributors
 */
#pragma once

#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr

#include "../common/io.h"        // for AlignedResourceReadStream
#include "sparse_page_writer.h"  // for SparsePageFormat
#include "xgboost/data.h"        // for EllpackPage

namespace xgboost::common {
class HistogramCuts;
}

namespace xgboost::data {
class EllpackPageRawFormat : public SparsePageFormat<EllpackPage> {
  std::shared_ptr<common::HistogramCuts const> cuts_;

 public:
  explicit EllpackPageRawFormat(std::shared_ptr<common::HistogramCuts const> cuts);
  bool Read(EllpackPage* page, common::AlignedResourceReadStream* fi) override;
  std::size_t Write(const EllpackPage& page, common::AlignedFileWriteStream* fo) override;
};
}  // namespace xgboost::data
