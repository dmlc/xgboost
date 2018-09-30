/*!
 * Copyright 2015 by Contributors
 * \file simple_dmatrix.h
 * \brief In-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SIMPLE_DMATRIX_H_
#define XGBOOST_DATA_SIMPLE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include "simple_csr_source.h"

namespace xgboost {
namespace data {

class SimpleDMatrix : public DMatrix {
 public:
  explicit SimpleDMatrix(std::unique_ptr<DataSource>&& source)
      : source_(std::move(source)) {}

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  float GetColDensity(size_t cidx) override;

  bool SingleColBlock() const override;

  BatchSet GetRowBatches() override;

  BatchSet GetColumnBatches() override;

  BatchSet GetSortedColumnBatches() override;

 private:
  // source data pointer.
  std::unique_ptr<DataSource> source_;

  std::unique_ptr<SparsePage> sorted_column_page_;
  std::unique_ptr<SparsePage> column_page_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_H_
