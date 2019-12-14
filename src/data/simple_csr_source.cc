/*!
 * Copyright 2015-2019 by Contributors
 * \file simple_csr_source.cc
 */
#include <dmlc/base.h>
#include <xgboost/logging.h>
#include <xgboost/json.h>

#include "simple_csr_source.h"
#include "columnar.h"

namespace xgboost {
namespace data {

void SimpleCSRSource::Clear() {
  page_.Clear();
  this->info.Clear();
}

void SimpleCSRSource::CopyFrom(DMatrix* src) {
  this->Clear();
  this->info = src->Info();
  for (const auto &batch : src->GetBatches<SparsePage>()) {
    page_.Push(batch);
  }
}

void SimpleCSRSource::LoadBinary(dmlc::Stream* fi) {
  int tmagic;
  CHECK(fi->Read(&tmagic, sizeof(tmagic)) == sizeof(tmagic)) << "invalid input file format";
  CHECK_EQ(tmagic, kMagic) << "invalid format, magic number mismatch";
  info.LoadBinary(fi);
  fi->Read(&page_.offset.HostVector());
  fi->Read(&page_.data.HostVector());
}

void SimpleCSRSource::SaveBinary(dmlc::Stream* fo) const {
  int tmagic = kMagic;
  fo->Write(&tmagic, sizeof(tmagic));
  info.SaveBinary(fo);
  fo->Write(page_.offset.HostVector());
  fo->Write(page_.data.HostVector());
}

void SimpleCSRSource::BeforeFirst() {
  at_first_ = true;
}

bool SimpleCSRSource::Next() {
  if (!at_first_) return false;
  at_first_ = false;
  return true;
}

const SparsePage& SimpleCSRSource::Value() const {
  return page_;
}

/*!
 * Please be careful that, in official specification, the only three required fields are
 * `shape', `version' and `typestr'.  Any other is optional, including `data'.  But here
 * we have one additional requirements for input data:
 *
 * - `data' field is required, passing in an empty dataset is not accepted, as most (if
 *   not all) of our algorithms don't have test for empty dataset.  An error is better
 *   than a crash.
 *
 * Missing value handling:
 *   Missing value is specified:
 *     - Ignore the validity mask from columnar format.
 *     - Remove entries that equals to missing value.
 *     - missing = NaN:
 *        - Remove entries that is NaN
 *     - missing != NaN:
 *        - Check for NaN entries, throw an error if found.
 *   Missing value is not specified:
 *     - Remove entries that is specifed as by validity mask.
 *     - Remove NaN entries.
 *
 * What if invalid value from dataframe is 0 but I specify missing=NaN in XGBoost?  Since
 * validity mask is ignored, all 0s are preserved in XGBoost.
 *
 * FIXME(trivialfis): Put above into document after we have a consistent way for
 * processing input data.
 *
 * Sample input:
 * [
 *   {
 *     "shape": [
 *       10
 *     ],
 *     "strides": [
 *       4
 *     ],
 *     "data": [
 *       30074864128,
 *       false
 *     ],
 *     "typestr": "<f4",
 *     "version": 1,
 *     "mask": {
 *       "shape": [
 *         64
 *       ],
 *       "strides": [
 *         1
 *       ],
 *       "data": [
 *         30074864640,
 *         false
 *       ],
 *       "typestr": "|i1",
 *       "version": 1
 *     }
 *   }
 * ]
 */
void SimpleCSRSource::CopyFrom(std::string const& cuda_interfaces_str,
                               bool has_missing, float missing) {
  Json interfaces = Json::Load({cuda_interfaces_str.c_str(),
                                cuda_interfaces_str.size()});
  std::vector<Json> const& columns = get<Array>(interfaces);
  size_t n_columns = columns.size();
  CHECK_GT(n_columns, 0) << "Number of columns must not eqaul to 0.";

  auto const& typestr = get<String const>(columns[0]["typestr"]);
  CHECK_EQ(typestr.size(),    3)  << ColumnarErrors::TypestrFormat();
  CHECK_NE(typestr.front(), '>')  << ColumnarErrors::BigEndian();

  this->FromDeviceColumnar(columns, has_missing, missing);
}

#if !defined(XGBOOST_USE_CUDA)
void SimpleCSRSource::FromDeviceColumnar(std::vector<Json> const& columns,
                                         bool has_missing, float missing) {
  LOG(FATAL) << "XGBoost version is not compiled with GPU support";
}
#endif  // !defined(XGBOOST_USE_CUDA)

}  // namespace data
}  // namespace xgboost
