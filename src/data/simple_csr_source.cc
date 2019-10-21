/*!
 * Copyright 2015-2019 by Contributors
 * \file simple_csr_source.cc
 */
#include <dmlc/base.h>
#include <dmlc/data.h>  // dmlc::RowBlock

#include <xgboost/logging.h>
#include <xgboost/json.h>

#include <limits>
#include "simple_csr_source.h"
#include "columnar.h"
#include "../common/math.h"

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

// There is no validity mask here so we don't care if NaN is default value or is specifed
// by user.
void SimpleCSRSource::CopyFrom(dmlc::Parser<uint32_t>* parser, float const missing) {
  // use qid to get group info
  const uint64_t default_max = std::numeric_limits<uint64_t>::max();
  uint64_t last_group_id = default_max;
  bst_uint group_size = 0;
  this->Clear();
  while (parser->Next()) {
    // Setup MetaInfo
    const dmlc::RowBlock<uint32_t>& batch = parser->Value();
    if (batch.label != nullptr) {
      auto& labels = info.labels_.HostVector();
      labels.insert(labels.end(), batch.label, batch.label + batch.size);
    }
    if (batch.weight != nullptr) {
      auto& weights = info.weights_.HostVector();
      weights.insert(weights.end(), batch.weight, batch.weight + batch.size);
    }
    if (batch.qid != nullptr) {
      // get group
      for (size_t i = 0; i < batch.size; ++i) {
        const uint64_t cur_group_id = batch.qid[i];
        if (last_group_id == default_max || last_group_id != cur_group_id) {
          info.group_ptr_.push_back(group_size);
        }
        last_group_id = cur_group_id;
        ++group_size;
      }
    }

    // Remove the assertion on batch.index, which can be null in the case that the data in this
    // batch is entirely sparse. Although it's true that this indicates a likely issue with the
    // user's data workflows, passing XGBoost entirely sparse data should not cause it to fail.
    // See https://github.com/dmlc/xgboost/issues/1827 for complete detail.
    // CHECK(batch.index != nullptr);

    bool const nan_missing = common::CheckNAN(missing);
    // update information
    this->info.num_row_ += batch.size;
    // copy the data over
    auto& data_vec = page_.data.HostVector();
    auto& offset_vec = page_.offset.HostVector();

    // current row offset with missing value removed
    size_t offset_without_missing { batch.offset[0] };

    for (size_t row_id = 0; row_id  < batch.size; ++row_id) {
      for (size_t i = batch.offset[row_id]; i < batch.offset[row_id+1]; ++i) {
        uint32_t const index = batch.index[i];
        float const fvalue = batch.value == nullptr ? 1.0f : batch.value[i];

        if (XGBOOST_EXPECT((!nan_missing && common::CheckNAN(fvalue)), false)) {
          LOG(FATAL) << "There are NAN in the matrix, however, but missing is set to: " << missing;
        }
        if (common::CheckNAN(fvalue) || batch.value[i] == missing) {
          // found a missing value
          continue;
        }

        offset_without_missing += 1;
        data_vec.emplace_back(index, fvalue);
        this->info.num_col_ =
            std::max(this->info.num_col_, static_cast<uint64_t>(index + 1));
      }

      offset_vec.push_back(offset_without_missing);
    }

    // for (size_t i = batch.offset[0]; i < batch.offset[batch.size]; ++i) {
    //   uint32_t const index = batch.index[i];
    //   float const fvalue = batch.value == nullptr ? 1.0f : batch.value[i];
    //   if ((nan_missing && common::CheckNAN(fvalue)) || batch.value[i] == missing) {
    //     // found a missing value
    //     continue;
    //   }
    //   offset_without_missing += 1;
    //   data_vec.emplace_back(index, fvalue);
    //   this->info.num_col_ = std::max(this->info.num_col_,
    //                                  static_cast<uint64_t>(index + 1));
    // }
    // size_t top = page_.offset.Size();
    // for (size_t i = 0; i < batch.size; ++i) {
    //   offset_vec.push_back(offset_vec[top - 1] + batch.offset[i + 1] - batch.offset[0]);
    // }
  }

  if (last_group_id != default_max) {
    if (group_size > info.group_ptr_.back()) {
      info.group_ptr_.push_back(group_size);
    }
  }

  auto& offset_vec = page_.offset.HostVector();
  this->info.num_nonzero_ = static_cast<uint64_t>(page_.data.Size());
  CHECK(info.group_ptr_.size() == 0 || info.group_ptr_.back() == offset_vec.back());
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
                               bool has_missing,
                               float const missing) {
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
