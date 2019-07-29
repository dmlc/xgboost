/*!
 * Copyright 2015 by Contributors
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
      info.qids_.insert(info.qids_.end(), batch.qid, batch.qid + batch.size);
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
  CHECK(info.qids_.empty() || info.qids_.size() == info.num_row_);
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

/*
[
  {
    "shape": [
      10
    ],
    "strides": [
      4
    ],
    "data": [
      30074864128,
      false
    ],
    "typestr": "<f4",
    "version": 1,
    "mask": {
      "shape": [
        64
      ],
      "strides": [
        1
      ],
      "data": [
        30074864640,
        false
      ],
      "typestr": "|i1",
      "version": 1,
      "null_count": 1
    }
  }
]
 */
void SimpleCSRSource::CopyFrom(std::vector<Json> columns) {
  size_t n_columns = columns.size();
  std::vector<ForeignColumn> foreign_cols(n_columns);
  for (size_t i = 0; i < columns.size(); ++i) {
    CHECK(IsA<Object>(columns[i]));

    auto const& column = get<Object const>(columns[i]);

    // Find null mask (validity mask) field
    common::Span<BitField::value_type> s_mask;
    long null_count {0};
    if (column.find("mask") != column.cend()) {
      auto const& j_mask = get<Object const>(column.at("mask"));
      auto p_mask =
          reinterpret_cast<BitField::value_type*>(static_cast<size_t>(
              get<Integer const>(
                  get<Array const>(
                      j_mask.at("data"))
                  .at(0))));

      auto j_shape = get<Array const>(j_mask.at("shape"));
      CHECK_EQ(j_shape.size(), 1) << "Mask should be an 1 dimension array.";
      CHECK_EQ(get<Integer>(j_shape.front()) % 8, 0) <<
          "Length of validity map must be a multiple of 8 bytes.";
      int64_t size = get<Integer>(j_shape.at(0)) * sizeof(unsigned char) / sizeof(BitField::value_type);
      s_mask = {p_mask, size};

      CHECK(j_mask.find("null_count") != j_mask.cend()) <<
          "Column with null mask must include null_count as "
          "part of mask object for XGBoost.";
      null_count = get<Integer const>(j_mask.at("null_count"));
      LOG(CONSOLE) << "mask: " << __LINE__ << (size_t)p_mask;
    }

    // Find data field
    float* p_data = reinterpret_cast<float*>(static_cast<size_t>(
        get<Integer const>(
            get<Array const>(
                column.at("data")).at(0))));

    auto version = get<Integer const>(column.at("version"));
    CHECK_EQ(version, 1) << "Only version 1 of __cuda_array_interface__ is being supported";

    auto typestr = get<String const>(column.at("typestr"));
    CHECK_EQ(typestr.size(),    3) << "`typestr` should be of format <endian><type><size>.";
    CHECK_NE(typestr.front(), '>') << "Big endian is not supported yet.";
    CHECK_EQ(typestr.at(1),   'f') << "Data should be of floating point type.";
    CHECK_EQ(typestr.at(2),   '4') << "Please convert the input into float32 first.";

    auto strides = get<Array const>(column.at("strides"));
    CHECK_EQ(strides.size(), 1)              << "Only 1 dimensional array is valid.";
    CHECK_EQ(get<Integer>(strides.at(0)), 4) << "Memory should be contigious.";

    auto j_shape = get<Array const>(column.at("shape"));
    CHECK_EQ(j_shape.size(), 1) << "Only 1 dimension column is valid.";
    auto length = get<Integer const>(j_shape.at(0));

    common::Span<float> s_data {p_data, length};

    foreign_cols[i].data  = s_data;
    foreign_cols[i].valid = BitField(s_mask);
    foreign_cols[i].size  = s_data.size();
    foreign_cols[i].null_count = null_count;
  }

  this->CopyFrom(foreign_cols);
}

}  // namespace data
}  // namespace xgboost
