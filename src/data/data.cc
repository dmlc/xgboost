/*!
 * Copyright 2015 by Contributors
 * \file data.cc
 */
#include <xgboost/data.h>
#include <cstring>

namespace xgboost {
// implementation of inline functions
void MetaInfo::Clear() {
  num_row = num_col = num_nonzero = 0;
  labels.clear();
  root_index.clear();
  group_ptr.clear();
  weights.clear();
  base_margin.clear();
}

void MetaInfo::SaveBinary(dmlc::Stream *fo) const {
  int version = kVersion;
  fo->Write(&version, sizeof(version));
  fo->Write(&num_row, sizeof(num_row));
  fo->Write(&num_col, sizeof(num_col));
  fo->Write(&num_nonzero, sizeof(num_nonzero));
  fo->Write(labels);
  fo->Write(group_ptr);
  fo->Write(weights);
  fo->Write(root_index);
  fo->Write(base_margin);
}

void MetaInfo::LoadBinary(dmlc::Stream *fi) {
  int version;
  CHECK(fi->Read(&version, sizeof(version)) == sizeof(version)) << "MetaInfo: invalid version";
  CHECK_EQ(version, kVersion) << "MetaInfo: invalid format";
  CHECK(fi->Read(&num_row, sizeof(num_row)) == sizeof(num_row)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&num_col, sizeof(num_col)) == sizeof(num_col)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&num_nonzero, sizeof(num_nonzero)) == sizeof(num_nonzero))
      << "MetaInfo: invalid format";
  CHECK(fi->Read(&labels)) <<  "MetaInfo: invalid format";
  CHECK(fi->Read(&group_ptr)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&weights)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&root_index)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&base_margin)) << "MetaInfo: invalid format";
}

// macro to dispatch according to specified pointer types
#define DISPATCH_CONST_PTR(dtype, old_ptr, cast_ptr, proc)              \
  switch (dtype) {                                                      \
    case kFloat32: {                                                    \
      const float* cast_ptr = reinterpret_cast<const float*>(old_ptr); proc; break; \
    }                                                                   \
    case kDouble: {                                                     \
      const double* cast_ptr = reinterpret_cast<const double*>(old_ptr); proc; break; \
    }                                                                   \
    case kUInt32: {                                                     \
      const uint32_t* cast_ptr = reinterpret_cast<const uint32_t*>(old_ptr); proc; break; \
    }                                                                   \
    case kUInt64: {                                                     \
      const uint64_t* cast_ptr = reinterpret_cast<const uint64_t*>(old_ptr); proc; break; \
    }                                                                   \
    default: LOG(FATAL) << "Unknown data type" << dtype;                \
  }                                                                     \


void MetaInfo::SetInfo(const char* key, const void* dptr, DataType dtype, size_t num) {
  if (!std::strcmp(key, "root_index")) {
    root_index.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, root_index.begin()));
  } else if (!std::strcmp(key, "label")) {
    labels.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, labels.begin()));
  } else if (!std::strcmp(key, "weight")) {
    weights.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, weights.begin()));
  } else if (!std::strcmp(key, "base_margin")) {
    base_margin.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, base_margin.begin()));
  }
}

}  // namespace xgboost
