/*!
 * Copyright 2015 by Contributors
 * \file data.cc
 */
#include <xgboost/data.h>
#include <xgboost/logging.h>
#include <dmlc/registry.h>
#include <cstring>
#include "./sparse_page_writer.h"
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
  num_row_ = num_col_ = num_nonzero_ = 0;
  labels_.clear();
  root_index_.clear();
  group_ptr_.clear();
  qids_.clear();
  weights_.clear();
  base_margin_.clear();
}

void MetaInfo::SaveBinary(dmlc::Stream *fo) const {
  int32_t version = kVersion;
  fo->Write(&version, sizeof(version));
  fo->Write(&num_row_, sizeof(num_row_));
  fo->Write(&num_col_, sizeof(num_col_));
  fo->Write(&num_nonzero_, sizeof(num_nonzero_));
  fo->Write(labels_);
  fo->Write(group_ptr_);
  fo->Write(qids_);
  fo->Write(weights_);
  fo->Write(root_index_);
  fo->Write(base_margin_);
}

void MetaInfo::LoadBinary(dmlc::Stream *fi) {
  int version;
  CHECK(fi->Read(&version, sizeof(version)) == sizeof(version)) << "MetaInfo: invalid version";
  CHECK(version >= 1 && version <= kVersion) << "MetaInfo: unsupported file version";
  CHECK(fi->Read(&num_row_, sizeof(num_row_)) == sizeof(num_row_)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&num_col_, sizeof(num_col_)) == sizeof(num_col_)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&num_nonzero_, sizeof(num_nonzero_)) == sizeof(num_nonzero_))
      << "MetaInfo: invalid format";
  CHECK(fi->Read(&labels_)) <<  "MetaInfo: invalid format";
  CHECK(fi->Read(&group_ptr_)) << "MetaInfo: invalid format";
  if (version >= kVersionQidAdded) {
    CHECK(fi->Read(&qids_)) << "MetaInfo: invalid format";
  } else {  // old format doesn't contain qid field
    qids_.clear();
  }
  CHECK(fi->Read(&weights_)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&root_index_)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&base_margin_)) << "MetaInfo: invalid format";
}

// try to load group information from file, if exists
inline bool MetaTryLoadGroup(const std::string& fname,
                             std::vector<unsigned>* group) {
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r", true));
  if (fi == nullptr) return false;
  dmlc::istream is(fi.get());
  group->clear();
  group->push_back(0);
  unsigned nline = 0;
  while (is >> nline) {
    group->push_back(group->back() + nline);
  }
  return true;
}

// try to load weight information from file, if exists
inline bool MetaTryLoadFloatInfo(const std::string& fname,
                                 std::vector<bst_float>* data) {
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r", true));
  if (fi == nullptr) return false;
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
      auto cast_ptr = reinterpret_cast<const float*>(old_ptr); proc; break; \
    }                                                                   \
    case kDouble: {                                                     \
      auto cast_ptr = reinterpret_cast<const double*>(old_ptr); proc; break; \
    }                                                                   \
    case kUInt32: {                                                     \
      auto cast_ptr = reinterpret_cast<const uint32_t*>(old_ptr); proc; break; \
    }                                                                   \
    case kUInt64: {                                                     \
      auto cast_ptr = reinterpret_cast<const uint64_t*>(old_ptr); proc; break; \
    }                                                                   \
    default: LOG(FATAL) << "Unknown data type" << dtype;                \
  }                                                                     \


void MetaInfo::SetInfo(const char* key, const void* dptr, DataType dtype, size_t num) {
  if (!std::strcmp(key, "root_index")) {
    root_index_.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, root_index_.begin()));
  } else if (!std::strcmp(key, "label")) {
    labels_.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, labels_.begin()));
  } else if (!std::strcmp(key, "weight")) {
    weights_.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, weights_.begin()));
  } else if (!std::strcmp(key, "base_margin")) {
    base_margin_.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, base_margin_.begin()));
  } else if (!std::strcmp(key, "group")) {
    group_ptr_.resize(num + 1);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, group_ptr_.begin() + 1));
    group_ptr_[0] = 0;
    for (size_t i = 1; i < group_ptr_.size(); ++i) {
      group_ptr_[i] = group_ptr_[i - 1] + group_ptr_[i];
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
        if (i + 1 != cache_shards.size()) {
          os << ':';
        }
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
    if (fi != nullptr) {
      common::PeekableInStream is(fi.get());
      if (is.PeekRead(&magic, sizeof(magic)) == sizeof(magic) &&
          magic == data::SimpleCSRSource::kMagic) {
        std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());
        source->LoadBinary(&is);
        DMatrix* dmat = DMatrix::Create(std::move(source), cache_file);
        if (!silent) {
          LOG(CONSOLE) << dmat->Info().num_row_ << 'x' << dmat->Info().num_col_ << " matrix with "
                       << dmat->Info().num_nonzero_ << " entries loaded from " << uri;
        }
        return dmat;
      }
    }
  }

  std::unique_ptr<dmlc::Parser<uint32_t> > parser(
      dmlc::Parser<uint32_t>::Create(fname.c_str(), partid, npart, file_format.c_str()));
  DMatrix* dmat = DMatrix::Create(parser.get(), cache_file);
  if (!silent) {
    LOG(CONSOLE) << dmat->Info().num_row_ << 'x' << dmat->Info().num_col_ << " matrix with "
                 << dmat->Info().num_nonzero_ << " entries loaded from " << uri;
  }
  /* sync up number of features after matrix loaded.
   * partitioned data will fail the train/val validation check 
   * since partitioned data not knowing the real number of features. */
  rabit::Allreduce<rabit::op::Max>(&dmat->Info().num_col_, 1);
  // backward compatiblity code.
  if (!load_row_split) {
    MetaInfo& info = dmat->Info();
    if (MetaTryLoadGroup(fname + ".group", &info.group_ptr_) && !silent) {
      LOG(CONSOLE) << info.group_ptr_.size() - 1
                   << " groups are loaded from " << fname << ".group";
    }
    if (MetaTryLoadFloatInfo(fname + ".base_margin", &info.base_margin_) && !silent) {
      LOG(CONSOLE) << info.base_margin_.size()
                   << " base_margin are loaded from " << fname << ".base_margin";
    }
    if (MetaTryLoadFloatInfo(fname + ".weight", &info.weights_) && !silent) {
      LOG(CONSOLE) << info.weights_.size()
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
  data::SparsePageFormat* data::SparsePageFormat::Create(const std::string& name) {
  auto *e = ::dmlc::Registry< ::xgboost::data::SparsePageFormatReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown format type " << name;
  }
  return (e->body)();
}

std::pair<std::string, std::string>
data::SparsePageFormat::DecideFormat(const std::string& cache_prefix) {
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

namespace data {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(sparse_page_raw_format);
}  // namespace data
}  // namespace xgboost
