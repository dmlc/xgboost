/*!
 * Copyright 2015-2019 by Contributors
 * \file simple_csr_source.cc
 */
#include <dmlc/base.h>
#include <xgboost/logging.h>
#include <xgboost/json.h>

#include <limits>
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

void SimpleCSRSource::CopyFrom(dmlc::Parser<uint32_t>* parser) {
  // use qid to get group info
  const uint64_t default_max = std::numeric_limits<uint64_t>::max();
  uint64_t last_group_id = default_max;
  bst_uint group_size = 0;
  std::vector<uint64_t> qids;
  this->Clear();
  while (parser->Next()) {
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
      qids.insert(qids.end(), batch.qid, batch.qid + batch.size);
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

    // update information
    this->info.num_row_ += batch.size;
    // copy the data over
    auto& data_vec = page_.data.HostVector();
    auto& offset_vec = page_.offset.HostVector();
    for (size_t i = batch.offset[0]; i < batch.offset[batch.size]; ++i) {
      uint32_t index = batch.index[i];
      bst_float fvalue = batch.value == nullptr ? 1.0f : batch.value[i];
      data_vec.emplace_back(index, fvalue);
      this->info.num_col_ = std::max(this->info.num_col_,
                                    static_cast<uint64_t>(index + 1));
    }
    size_t top = page_.offset.Size();
    for (size_t i = 0; i < batch.size; ++i) {
      offset_vec.push_back(offset_vec[top - 1] + batch.offset[i + 1] - batch.offset[0]);
    }
  }
  if (last_group_id != default_max) {
    if (group_size > info.group_ptr_.back()) {
      info.group_ptr_.push_back(group_size);
    }
  }
  this->info.num_nonzero_ = static_cast<uint64_t>(page_.data.Size());
  // Either every row has query ID or none at all
  CHECK(qids.empty() || qids.size() == info.num_row_);
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
 * we have two additional requirements for input data:
 *
 * - `data' field is required, passing in an empty dataset is not accepted, as most (if
 *   not all) of our algorithms don't have test for empty dataset.  An error is better
 *   than a crash.
 *
 * - `null_count' is required when `mask' is presented.  We can compute `null_count'
 *   ourselves and copy the result back to host for memory allocation.  But it's in the
 *   specification of Apache Arrow hence it should be readily available,
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
 *       "version": 1,
 *       "null_count": 1
 *     }
 *   }
 * ]
 */
void SimpleCSRSource::CopyFrom(std::string const& cuda_interfaces_str) {
  Json interfaces = Json::Load({cuda_interfaces_str.c_str(),
                                cuda_interfaces_str.size()});
  std::vector<Json> const& columns = get<Array>(interfaces);
  size_t n_columns = columns.size();
  CHECK_GT(n_columns, 0);

  std::vector<Columnar> foreign_cols(n_columns);
  for (size_t i = 0; i < columns.size(); ++i) {
    CHECK(IsA<Object>(columns[i]));
    auto const& column = get<Object const>(columns[i]);

    auto version = get<Integer const>(column.at("version"));
    CHECK_EQ(version, 1) << ColumnarErrors::Version();

    // Find null mask (validity mask) field
    // Mask object is also an array interface, but with different requirements.

    // TODO(trivialfis): Abstract this into a class that accept a json
    // object and turn it into an array (for cupy and numba).
    common::Span<RBitField8::value_type> s_mask;
    int32_t null_count {0};
    if (column.find("mask") != column.cend()) {
      auto const& j_mask = get<Object const>(column.at("mask"));
      auto p_mask = GetPtrFromArrayData<RBitField8::value_type*>(j_mask);

      auto j_shape = get<Array const>(j_mask.at("shape"));
      CHECK_EQ(j_shape.size(), 1) << ColumnarErrors::Dimension(1);
      CHECK_EQ(get<Integer>(j_shape.front()) % 8, 0) <<
          "Length of validity map must be a multiple of 8 bytes.";
      int64_t size = get<Integer>(j_shape.at(0)) *
                     sizeof(unsigned char) / sizeof(RBitField8::value_type);
      s_mask = {p_mask, size};
      auto typestr = get<String const>(j_mask.at("typestr"));
      CHECK_EQ(typestr.size(),    3) << ColumnarErrors::TypestrFormat();
      CHECK_NE(typestr.front(), '>') << ColumnarErrors::BigEndian();
      CHECK_EQ(typestr.at(1),   'i') << "mask" << ColumnarErrors::ofType("unsigned char");
      CHECK_EQ(typestr.at(2),   '1') << "mask" << ColumnarErrors::toUInt();

      CHECK(j_mask.find("null_count") != j_mask.cend()) <<
          "Column with null mask must include null_count as "
          "part of mask object for XGBoost.";
      null_count = get<Integer const>(j_mask.at("null_count"));
    }

    // Find data field
    if (column.find("data") == column.cend()) {
      LOG(FATAL) << "Empty dataset passed in.";
    }

    auto typestr = get<String const>(column.at("typestr"));
    CHECK_EQ(typestr.size(),    3) << ColumnarErrors::TypestrFormat();
    CHECK_NE(typestr.front(), '>') << ColumnarErrors::BigEndian();
    CHECK_EQ(typestr.at(1),   'f') << "data" << ColumnarErrors::ofType("floating point");
    CHECK_EQ(typestr.at(2),   '4') << ColumnarErrors::toFloat();

    auto j_shape = get<Array const>(column.at("shape"));
    CHECK_EQ(j_shape.size(), 1) << ColumnarErrors::Dimension(1);

    if (column.find("strides") != column.cend()) {
      auto strides = get<Array const>(column.at("strides"));
      CHECK_EQ(strides.size(), 1)              << ColumnarErrors::Dimension(1);
      CHECK_EQ(get<Integer>(strides.at(0)), 4) << ColumnarErrors::Contigious();
    }

    auto length = get<Integer const>(j_shape.at(0));

    float* p_data = GetPtrFromArrayData<float*>(column);
    common::Span<float> s_data {p_data, length};

    foreign_cols[i].data  = s_data;
    foreign_cols[i].valid = RBitField8(s_mask);
    foreign_cols[i].size  = s_data.size();
    foreign_cols[i].null_count = null_count;
  }

  info.num_col_ = n_columns;
  info.num_row_ = foreign_cols[0].size;
  for (size_t i = 0; i < n_columns; ++i) {
    CHECK_EQ(foreign_cols[0].size, foreign_cols[i].size);
    info.num_nonzero_ += foreign_cols[i].data.size() - foreign_cols[i].null_count;
  }

  this->FromDeviceColumnar(foreign_cols);
}

#if !defined(XGBOOST_USE_CUDA)
void SimpleCSRSource::FromDeviceColumnar(std::vector<Columnar> cols) {
  LOG(FATAL) << "XGBoost version is not compiled with GPU support";
}
#endif  // !defined(XGBOOST_USE_CUDA)

}  // namespace data
}  // namespace xgboost
