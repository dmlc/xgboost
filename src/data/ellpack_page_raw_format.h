/**
 * Copyright 2019-2024, XGBoost contributors
 */
#pragma once

#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr
#include <utility>  // for move

#include "../common/io.h"        // for AlignedResourceReadStream
#include "sparse_page_writer.h"  // for SparsePageFormat
#include "xgboost/data.h"        // for EllpackPage

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // for AssertGPUSupport
#endif                         // !defined(XGBOOST_USE_CUDA)`

namespace xgboost::common {
class HistogramCuts;
}

namespace xgboost::data {
class EllpackPageRawFormat : public SparsePageFormat<EllpackPage> {
  std::shared_ptr<common::HistogramCuts const> cuts_;

 public:
  explicit EllpackPageRawFormat(std::shared_ptr<common::HistogramCuts const> cuts)
      : cuts_{std::move(cuts)} {}
  [[nodiscard]] bool Read(EllpackPage* page, common::AlignedResourceReadStream* fi) override;
  [[nodiscard]] std::size_t Write(const EllpackPage& page,
                                  common::AlignedFileWriteStream* fo) override;
};

#if !defined(XGBOOST_USE_CUDA)
inline bool EllpackPageRawFormat::Read(EllpackPage*, common::AlignedResourceReadStream*) {
  common::AssertGPUSupport();
  return false;
}

inline std::size_t EllpackPageRawFormat::Write(const EllpackPage&,
                                               common::AlignedFileWriteStream*) {
  common::AssertGPUSupport();
  return 0;
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data
