/**
 *  Copyright 2023, XGBoost Contributors
 */
#include "sparse_page_source.h"

#include <unistd.h>  // for getpagesize

namespace xgboost::data {
std::size_t PadPageForMMAP(std::size_t file_bytes, dmlc::Stream* fo) {
  decltype(file_bytes) page_size = getpagesize();
  CHECK(page_size != 0 && page_size % 2 == 0) << "Failed to get page size on the current system.";
  CHECK_NE(file_bytes, 0) << "Empty page encountered.";
  auto n = file_bytes / page_size;
  auto padded = (n + !!(file_bytes % page_size != 0)) * page_size;
  auto padding = padded - file_bytes;
  std::vector<std::uint8_t> padding_bytes(padding, 0);
  fo->Write(padding_bytes.data(), padding_bytes.size());
  return padded;
}
}  // namespace xgboost::data
