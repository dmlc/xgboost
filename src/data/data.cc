/*!
 * Copyright 2015 by Contributors
 * \file data.cc
 */
#include <xgboost/data.h>
#include <xgboost/logging.h>
#include <cstring>
#include "./sparse_batch_page.h"
#include "./simple_dmatrix.h"
#include "./simple_csr_source.h"
#include "../common/io.h"

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


DMatrix* DMatrix::Load(const std::string& uri,
                       bool silent,
                       bool load_row_split,
                       const std::string& file_format) {
  std::string fname, cache_file;
  size_t dlm_pos = uri.find('#');
  if (dlm_pos != std::string::npos) {
    cache_file = uri.substr(dlm_pos + 1, uri.length());
    fname = uri.substr(0, dlm_pos);
    CHECK_EQ(cache_file.find('#'), std::string::npos)
        << "Only one `#` is allowed in file path for cache file specification.";
    if (load_row_split) {
      std::ostringstream os;
      os << cache_file << ".r" << rabit::GetRank();
      cache_file = os.str();
    }
  } else {
    fname = uri;
  }
  int partid = 0, npart = 1;
  if (load_row_split) {
    partid = rabit::GetRank();
    npart = rabit::GetWorldSize();
  }

  // legacy handling of binary data loading
  if (file_format == "auto" && !load_row_split) {
    int magic;
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r"));
    common::PeekableInStream is(fi.get());
     if (is.PeekRead(&magic, sizeof(magic)) == sizeof(magic) &&
         magic == data::SimpleCSRSource::kMagic) {
       std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());
       source->LoadBinary(&is);
       DMatrix* dmat = DMatrix::Create(std::move(source), cache_file);
       if (!silent) {
         LOG(CONSOLE) << dmat->info().num_row << 'x' << dmat->info().num_col << " matrix with "
                      << dmat->info().num_nonzero << " entries loaded from " << uri;
       }
       return dmat;
     }
  }

  std::string ftype = file_format;
  if (file_format == "auto") ftype = "libsvm";
  std::unique_ptr<dmlc::Parser<uint32_t> > parser(
      dmlc::Parser<uint32_t>::Create(fname.c_str(), partid, npart, ftype.c_str()));
  DMatrix* dmat = DMatrix::Create(parser.get(), cache_file);
  if (!silent) {
    LOG(CONSOLE) << dmat->info().num_row << 'x' << dmat->info().num_col << " matrix with "
                 << dmat->info().num_nonzero << " entries loaded from " << uri;
  }
  return dmat;
}

DMatrix* DMatrix::Create(dmlc::Parser<uint32_t>* parser,
                         const std::string& cache_prefix) {
  if (cache_prefix.length() == 0) {
    std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());
    source->CopyFrom(parser);
    return DMatrix::Create(std::move(source), cache_prefix);
  } else {
    LOG(FATAL) << "external memory not yet implemented";
    return nullptr;
  }
}

void DMatrix::SaveToLocalFile(const std::string& fname) {
  data::SimpleCSRSource source;
  source.CopyFrom(this);
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
  source.SaveBinary(fo.get());
}

DMatrix* DMatrix::Create(std::unique_ptr<DataSource>&& source,
                         const std::string& cache_prefix) {
  return new data::SimpleDMatrix(std::move(source));
}
}  // namespace xgboost
