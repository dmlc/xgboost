/*!
 * Copyright 2015 by Contributors
 * \file data.cc
 */
#include <xgboost/data.h>
#include <xgboost/logging.h>
#include <dmlc/registry.h>
#include <cstring>
#include "./sparse_batch_page.h"
#include "./simple_dmatrix.h"
#include "./simple_csr_source.h"
#include "../common/common.h"
#include "../common/io.h"

#if DMLC_ENABLE_STD_THREAD
#include "./sparse_page_source.h"
#include "./sparse_page_dmatrix.h"
#endif

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::data::SparsePageFormatReg);
}  // namespace dmlc

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
  int32_t version = kVersion;
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

// try to load group information from file, if exists
inline bool MetaTryLoadGroup(const std::string& fname,
                             std::vector<unsigned>* group) {
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r", true));
  if (fi.get() == nullptr) return false;
  dmlc::istream is(fi.get());
  group->clear();
  group->push_back(0);
  unsigned nline;
  while (is >> nline) {
    group->push_back(group->back() + nline);
  }
  return true;
}

// try to load weight information from file, if exists
inline bool MetaTryLoadFloatInfo(const std::string& fname,
                                 std::vector<bst_float>* data) {
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r", true));
  if (fi.get() == nullptr) return false;
  dmlc::istream is(fi.get());
  data->clear();
  bst_float value;
  while (is >> value) {
    data->push_back(value);
  }
  return true;
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
  } else if (!std::strcmp(key, "group")) {
    group_ptr.resize(num + 1);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, group_ptr.begin() + 1));
    group_ptr[0] = 0;
    for (size_t i = 1; i < group_ptr.size(); ++i) {
      group_ptr[i] = group_ptr[i - 1] + group_ptr[i];
    }
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
      std::vector<std::string> cache_shards = common::Split(cache_file, ':');
      for (size_t i = 0; i < cache_shards.size(); ++i) {
        size_t pos = cache_shards[i].rfind('.');
        if (pos == std::string::npos) {
          os << cache_shards[i]
             << ".r" << rabit::GetRank()
             << "-" <<  rabit::GetWorldSize();
        } else {
          os << cache_shards[i].substr(0, pos)
             << ".r" << rabit::GetRank()
             << "-" <<  rabit::GetWorldSize()
             << cache_shards[i].substr(pos, cache_shards[i].length());
        }
        if (i + 1 != cache_shards.size()) os << ':';
      }
      cache_file = os.str();
    }
  } else {
    fname = uri;
  }
  int partid = 0, npart = 1;
  if (load_row_split) {
    partid = rabit::GetRank();
    npart = rabit::GetWorldSize();
  } else {
    // test option to load in part
    npart = dmlc::GetEnv("XGBOOST_TEST_NPART", 1);
  }

  if (npart != 1) {
    LOG(CONSOLE) << "Load part of data " << partid
                 << " of " << npart << " parts";
  }
  // legacy handling of binary data loading
  if (file_format == "auto" && npart == 1) {
    int magic;
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r", true));
    if (fi.get() != nullptr) {
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
  }

  std::unique_ptr<dmlc::Parser<uint32_t> > parser(
      dmlc::Parser<uint32_t>::Create(fname.c_str(), partid, npart, file_format.c_str()));
  DMatrix* dmat = DMatrix::Create(parser.get(), cache_file);
  if (!silent) {
    LOG(CONSOLE) << dmat->info().num_row << 'x' << dmat->info().num_col << " matrix with "
                 << dmat->info().num_nonzero << " entries loaded from " << uri;
  }
  // backward compatiblity code.
  if (!load_row_split) {
    MetaInfo& info = dmat->info();
    if (MetaTryLoadGroup(fname + ".group", &info.group_ptr) && !silent) {
      LOG(CONSOLE) << info.group_ptr.size() - 1
                   << " groups are loaded from " << fname << ".group";
    }
    if (MetaTryLoadFloatInfo(fname + ".base_margin", &info.base_margin) && !silent) {
      LOG(CONSOLE) << info.base_margin.size()
                   << " base_margin are loaded from " << fname << ".base_margin";
    }
    if (MetaTryLoadFloatInfo(fname + ".weight", &info.weights) && !silent) {
      LOG(CONSOLE) << info.weights.size()
                   << " weights are loaded from " << fname << ".weight";
    }
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
#if DMLC_ENABLE_STD_THREAD
    if (!data::SparsePageSource::CacheExist(cache_prefix)) {
      data::SparsePageSource::Create(parser, cache_prefix);
    }
    std::unique_ptr<data::SparsePageSource> source(new data::SparsePageSource(cache_prefix));
    return DMatrix::Create(std::move(source), cache_prefix);
#else
    LOG(FATAL) << "External memory is not enabled in mingw";
    return nullptr;
#endif
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
  if (cache_prefix.length() == 0) {
    return new data::SimpleDMatrix(std::move(source));
  } else {
#if DMLC_ENABLE_STD_THREAD
    return new data::SparsePageDMatrix(std::move(source), cache_prefix);
#else
    LOG(FATAL) << "External memory is not enabled in mingw";
    return nullptr;
#endif
  }
}
}  // namespace xgboost

namespace xgboost {
namespace data {
SparsePage::Format* SparsePage::Format::Create(const std::string& name) {
  auto *e = ::dmlc::Registry< ::xgboost::data::SparsePageFormatReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown format type " << name;
  }
  return (e->body)();
}

std::pair<std::string, std::string>
SparsePage::Format::DecideFormat(const std::string& cache_prefix) {
  size_t pos = cache_prefix.rfind(".fmt-");

  if (pos != std::string::npos) {
    std::string fmt = cache_prefix.substr(pos + 5, cache_prefix.length());
    size_t cpos = fmt.rfind('-');
    if (cpos != std::string::npos) {
      return std::make_pair(fmt.substr(0, cpos), fmt.substr(cpos + 1, fmt.length()));
    } else {
      return std::make_pair(fmt, fmt);
    }
  } else {
    std::string raw = "raw";
    return std::make_pair(raw, raw);
  }
}

// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(sparse_page_raw_format);
}  // namespace data
}  // namespace xgboost
