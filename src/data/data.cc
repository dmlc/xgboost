/*!
 * Copyright 2015-2020 by Contributors
 * \file data.cc
 */
#include <dmlc/registry.h>
#include <cstring>

#include "dmlc/io.h"
#include "xgboost/data.h"
#include "xgboost/c_api.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/logging.h"
#include "xgboost/version_config.h"
#include "sparse_page_writer.h"
#include "simple_dmatrix.h"

#include "../common/io.h"
#include "../common/math.h"
#include "../common/version.h"
#include "../common/group_data.h"
#include "../data/adapter.h"

#if DMLC_ENABLE_STD_THREAD
#include "./sparse_page_source.h"
#include "./sparse_page_dmatrix.h"
#endif  // DMLC_ENABLE_STD_THREAD

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::data::SparsePageFormatReg<::xgboost::SparsePage>);
DMLC_REGISTRY_ENABLE(::xgboost::data::SparsePageFormatReg<::xgboost::CSCPage>);
DMLC_REGISTRY_ENABLE(::xgboost::data::SparsePageFormatReg<::xgboost::SortedCSCPage>);
DMLC_REGISTRY_ENABLE(::xgboost::data::SparsePageFormatReg<::xgboost::EllpackPage>);
}  // namespace dmlc

namespace {

template <typename T>
void SaveScalarField(dmlc::Stream *strm, const std::string &name,
                     xgboost::DataType type, const T &field) {
  strm->Write(name);
  strm->Write(static_cast<uint8_t>(type));
  strm->Write(true);  // is_scalar=True
  strm->Write(field);
}

template <typename T>
void SaveVectorField(dmlc::Stream *strm, const std::string &name,
                     xgboost::DataType type, std::pair<uint64_t, uint64_t> shape,
                     const std::vector<T>& field) {
  strm->Write(name);
  strm->Write(static_cast<uint8_t>(type));
  strm->Write(false);  // is_scalar=False
  strm->Write(shape.first);
  strm->Write(shape.second);
  strm->Write(field);
}

template <typename T>
void SaveVectorField(dmlc::Stream* strm, const std::string& name,
                     xgboost::DataType type, std::pair<uint64_t, uint64_t> shape,
                     const xgboost::HostDeviceVector<T>& field) {
  SaveVectorField(strm, name, type, shape, field.ConstHostVector());
}

template <typename T>
void LoadScalarField(dmlc::Stream* strm, const std::string& expected_name,
                     xgboost::DataType expected_type, T* field) {
  const std::string invalid {"MetaInfo: Invalid format. "};
  std::string name;
  xgboost::DataType type;
  bool is_scalar;
  CHECK(strm->Read(&name)) << invalid;
  CHECK_EQ(name, expected_name)
      << invalid << " Expected field: " << expected_name << ", got: " << name;
  uint8_t type_val;
  CHECK(strm->Read(&type_val)) << invalid;
  type = static_cast<xgboost::DataType>(type_val);
  CHECK(type == expected_type)
      << invalid << "Expected field of type: " << static_cast<int>(expected_type) << ", "
      << "got field type: " << static_cast<int>(type);
  CHECK(strm->Read(&is_scalar)) << invalid;
  CHECK(is_scalar)
    << invalid << "Expected field " << expected_name << " to be a scalar; got a vector";
  CHECK(strm->Read(field, sizeof(T))) << invalid;
}

template <typename T>
void LoadVectorField(dmlc::Stream* strm, const std::string& expected_name,
                     xgboost::DataType expected_type, std::vector<T>* field) {
  const std::string invalid {"MetaInfo: Invalid format. "};
  std::string name;
  xgboost::DataType type;
  bool is_scalar;
  CHECK(strm->Read(&name)) << invalid;
  CHECK_EQ(name, expected_name)
    << invalid << " Expected field: " << expected_name << ", got: " << name;
  uint8_t type_val;
  CHECK(strm->Read(&type_val)) << invalid;
  type = static_cast<xgboost::DataType>(type_val);
  CHECK(type == expected_type)
    << invalid << "Expected field of type: " << static_cast<int>(expected_type) << ", "
    << "got field type: " << static_cast<int>(type);
  CHECK(strm->Read(&is_scalar)) << invalid;
  CHECK(!is_scalar)
    << invalid << "Expected field " << expected_name << " to be a vector; got a scalar";
  std::pair<uint64_t, uint64_t> shape;

  CHECK(strm->Read(&shape.first));
  CHECK(strm->Read(&shape.second));
  // TODO(hcho3): this restriction may be lifted, once we add a field with more than 1 column.
  CHECK_EQ(shape.second, 1) << invalid << "Number of columns is expected to be 1.";

  CHECK(strm->Read(field)) << invalid;
}

template <typename T>
void LoadVectorField(dmlc::Stream* strm, const std::string& expected_name,
                     xgboost::DataType expected_type,
                     xgboost::HostDeviceVector<T>* field) {
  LoadVectorField(strm, expected_name, expected_type, &field->HostVector());
}

}  // anonymous namespace

namespace xgboost {

uint64_t constexpr MetaInfo::kNumField;

// implementation of inline functions
void MetaInfo::Clear() {
  num_row_ = num_col_ = num_nonzero_ = 0;
  labels_.HostVector().clear();
  group_ptr_.clear();
  weights_.HostVector().clear();
  base_margin_.HostVector().clear();
}

/*
 * Binary serialization format for MetaInfo:
 *
 * | name               | type     | is_scalar | num_row | num_col | value                   |
 * |--------------------+----------+-----------+---------+---------+-------------------------|
 * | num_row            | kUInt64  | True      | NA      |      NA | ${num_row_}             |
 * | num_col            | kUInt64  | True      | NA      |      NA | ${num_col_}             |
 * | num_nonzero        | kUInt64  | True      | NA      |      NA | ${num_nonzero_}         |
 * | labels             | kFloat32 | False     | ${size} |       1 | ${labels_}              |
 * | group_ptr          | kUInt32  | False     | ${size} |       1 | ${group_ptr_}           |
 * | weights            | kFloat32 | False     | ${size} |       1 | ${weights_}             |
 * | base_margin        | kFloat32 | False     | ${size} |       1 | ${base_margin_}         |
 * | labels_lower_bound | kFloat32 | False     | ${size} |       1 | ${labels_lower_bound__} |
 * | labels_upper_bound | kFloat32 | False     | ${size} |       1 | ${labels_upper_bound__} |
 *
 * Note that the scalar fields (is_scalar=True) will have num_row and num_col missing.
 * Also notice the difference between the saved name and the name used in `SetInfo':
 * the former uses the plural form.
 */

void MetaInfo::SaveBinary(dmlc::Stream *fo) const {
  Version::Save(fo);
  fo->Write(kNumField);
  int field_cnt = 0;  // make sure we are actually writing kNumField fields

  SaveScalarField(fo, u8"num_row", DataType::kUInt64, num_row_); ++field_cnt;
  SaveScalarField(fo, u8"num_col", DataType::kUInt64, num_col_); ++field_cnt;
  SaveScalarField(fo, u8"num_nonzero", DataType::kUInt64, num_nonzero_); ++field_cnt;
  SaveVectorField(fo, u8"labels", DataType::kFloat32,
                  {labels_.Size(), 1}, labels_); ++field_cnt;
  SaveVectorField(fo, u8"group_ptr", DataType::kUInt32,
                  {group_ptr_.size(), 1}, group_ptr_); ++field_cnt;
  SaveVectorField(fo, u8"weights", DataType::kFloat32,
                  {weights_.Size(), 1}, weights_); ++field_cnt;
  SaveVectorField(fo, u8"base_margin", DataType::kFloat32,
                  {base_margin_.Size(), 1}, base_margin_); ++field_cnt;
  SaveVectorField(fo, u8"labels_lower_bound", DataType::kFloat32,
                  {labels_lower_bound_.Size(), 1}, labels_lower_bound_); ++field_cnt;
  SaveVectorField(fo, u8"labels_upper_bound", DataType::kFloat32,
                  {labels_upper_bound_.Size(), 1}, labels_upper_bound_); ++field_cnt;

  CHECK_EQ(field_cnt, kNumField) << "Wrong number of fields";
}

void MetaInfo::LoadBinary(dmlc::Stream *fi) {
  auto version = Version::Load(fi);
  auto major = std::get<0>(version);
  // MetaInfo is saved in `SparsePageSource'.  So the version in MetaInfo represents the
  // version of DMatrix.
  CHECK_EQ(major, 1) << "Binary DMatrix generated by XGBoost: "
                     << Version::String(version) << " is no longer supported. "
                     << "Please process and save your data in current version: "
                     << Version::String(Version::Self()) << " again.";

  const uint64_t expected_num_field = kNumField;
  uint64_t num_field { 0 };
  CHECK(fi->Read(&num_field)) << "MetaInfo: invalid format";
  CHECK_GE(num_field, expected_num_field)
    << "MetaInfo: insufficient number of fields (expected at least " << expected_num_field
    << " fields, but the binary file only contains " << num_field << "fields.)";
  if (num_field > expected_num_field) {
    LOG(WARNING) << "MetaInfo: the given binary file contains extra fields which will be ignored.";
  }

  LoadScalarField(fi, u8"num_row", DataType::kUInt64, &num_row_);
  LoadScalarField(fi, u8"num_col", DataType::kUInt64, &num_col_);
  LoadScalarField(fi, u8"num_nonzero", DataType::kUInt64, &num_nonzero_);
  LoadVectorField(fi, u8"labels", DataType::kFloat32, &labels_);
  LoadVectorField(fi, u8"group_ptr", DataType::kUInt32, &group_ptr_);
  LoadVectorField(fi, u8"weights", DataType::kFloat32, &weights_);
  LoadVectorField(fi, u8"base_margin", DataType::kFloat32, &base_margin_);
  LoadVectorField(fi, u8"labels_lower_bound", DataType::kFloat32, &labels_lower_bound_);
  LoadVectorField(fi, u8"labels_upper_bound", DataType::kFloat32, &labels_upper_bound_);
}

template <typename T>
std::vector<T> Gather(const std::vector<T> &in, common::Span<int const> ridxs, size_t stride = 1) {
  if (in.empty()) {
    return {};
  }
  auto size = ridxs.size();
  std::vector<T> out(size * stride);
  for (auto i = 0ull; i < size; i++) {
    auto ridx = ridxs[i];
    for (size_t j = 0; j < stride; ++j) {
      out[i * stride +j] = in[ridx * stride + j];
    }
  }
  return out;
}

MetaInfo MetaInfo::Slice(common::Span<int32_t const> ridxs) const {
  MetaInfo out;
  out.num_row_ = ridxs.size();
  out.num_col_ = this->num_col_;
  // Groups is maintained by a higher level Python function.  We should aim at deprecating
  // the slice function.
  out.labels_.HostVector() = Gather(this->labels_.HostVector(), ridxs);
  out.labels_upper_bound_.HostVector() =
      Gather(this->labels_upper_bound_.HostVector(), ridxs);
  out.labels_lower_bound_.HostVector() =
      Gather(this->labels_lower_bound_.HostVector(), ridxs);
  // weights
  if (this->weights_.Size() + 1 == this->group_ptr_.size()) {
    auto& h_weights =  out.weights_.HostVector();
    // Assuming all groups are available.
    out.weights_.HostVector() = h_weights;
  } else {
    out.weights_.HostVector() = Gather(this->weights_.HostVector(), ridxs);
  }

  if (this->base_margin_.Size() != this->num_row_) {
    CHECK_EQ(this->base_margin_.Size() % this->num_row_, 0)
        << "Incorrect size of base margin vector.";
    size_t stride = this->base_margin_.Size() / this->num_row_;
    out.base_margin_.HostVector() = Gather(this->base_margin_.HostVector(), ridxs, stride);
  } else {
    out.base_margin_.HostVector() = Gather(this->base_margin_.HostVector(), ridxs);
  }
  return out;
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
    case xgboost::DataType::kFloat32: {                                 \
      auto cast_ptr = reinterpret_cast<const float*>(old_ptr); proc; break; \
    }                                                                   \
    case xgboost::DataType::kDouble: {                                  \
      auto cast_ptr = reinterpret_cast<const double*>(old_ptr); proc; break; \
    }                                                                   \
    case xgboost::DataType::kUInt32: {                                  \
      auto cast_ptr = reinterpret_cast<const uint32_t*>(old_ptr); proc; break; \
    }                                                                   \
    case xgboost::DataType::kUInt64: {                                  \
      auto cast_ptr = reinterpret_cast<const uint64_t*>(old_ptr); proc; break; \
    }                                                                   \
    default: LOG(FATAL) << "Unknown data type" << static_cast<uint8_t>(dtype); \
  }                                                                     \

void MetaInfo::SetInfo(const char* key, const void* dptr, DataType dtype, size_t num) {
  if (!std::strcmp(key, "label")) {
    auto& labels = labels_.HostVector();
    labels.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, labels.begin()));
  } else if (!std::strcmp(key, "weight")) {
    auto& weights = weights_.HostVector();
    weights.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, weights.begin()));
  } else if (!std::strcmp(key, "base_margin")) {
    auto& base_margin = base_margin_.HostVector();
    base_margin.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, base_margin.begin()));
  } else if (!std::strcmp(key, "group")) {
    group_ptr_.resize(num + 1);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, group_ptr_.begin() + 1));
    group_ptr_[0] = 0;
    for (size_t i = 1; i < group_ptr_.size(); ++i) {
      group_ptr_[i] = group_ptr_[i - 1] + group_ptr_[i];
    }
  } else if (!std::strcmp(key, "label_lower_bound")) {
    auto& labels = labels_lower_bound_.HostVector();
    labels.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, labels.begin()));
  } else if (!std::strcmp(key, "label_upper_bound")) {
    auto& labels = labels_upper_bound_.HostVector();
    labels.resize(num);
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,
                       std::copy(cast_dptr, cast_dptr + num, labels.begin()));
  } else {
    LOG(FATAL) << "Unknown key for MetaInfo: " << key;
  }
}

void MetaInfo::Validate(int32_t device) const {
  if (group_ptr_.size() != 0 && weights_.Size() != 0) {
    CHECK_EQ(group_ptr_.size(), weights_.Size() + 1)
        << "Size of weights must equal to number of groups when ranking "
           "group is used.";
    return;
  }
  if (group_ptr_.size() != 0) {
    CHECK_EQ(group_ptr_.back(), num_row_)
        << "Invalid group structure.  Number of rows obtained from groups "
           "doesn't equal to actual number of rows given by data.";
  }
  auto check_device = [device](HostDeviceVector<float> const &v) {
    CHECK(v.DeviceIdx() == GenericParameter::kCpuId ||
          device  == GenericParameter::kCpuId ||
          v.DeviceIdx() == device)
        << "Data is resided on a different device than `gpu_id`. "
        << "Device that data is on: " << v.DeviceIdx() << ", "
        << "`gpu_id` for XGBoost: " << device;
  };

  if (weights_.Size() != 0) {
    CHECK_EQ(weights_.Size(), num_row_)
        << "Size of weights must equal to number of rows.";
    check_device(weights_);
    return;
  }
  if (labels_.Size() != 0) {
    CHECK_EQ(labels_.Size(), num_row_)
        << "Size of labels must equal to number of rows.";
    check_device(labels_);
    return;
  }
  if (labels_lower_bound_.Size() != 0) {
    CHECK_EQ(labels_lower_bound_.Size(), num_row_)
        << "Size of label_lower_bound must equal to number of rows.";
    check_device(labels_lower_bound_);
    return;
  }
  if (labels_upper_bound_.Size() != 0) {
    CHECK_EQ(labels_upper_bound_.Size(), num_row_)
        << "Size of label_upper_bound must equal to number of rows.";
    check_device(labels_upper_bound_);
    return;
  }
  CHECK_LE(num_nonzero_, num_col_ * num_row_);
  if (base_margin_.Size() != 0) {
    CHECK_EQ(base_margin_.Size() % num_row_, 0)
        << "Size of base margin must be a multiple of number of rows.";
    check_device(base_margin_);
  }
}

#if !defined(XGBOOST_USE_CUDA)
void MetaInfo::SetInfo(const char * c_key, std::string const& interface_str) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)

DMatrix* DMatrix::Load(const std::string& uri,
                       bool silent,
                       bool load_row_split,
                       const std::string& file_format,
                       const size_t page_size) {
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
        magic == data::SimpleDMatrix::kMagic) {
        DMatrix* dmat = new data::SimpleDMatrix(&is);
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
  data::FileAdapter adapter(parser.get());
  DMatrix* dmat {nullptr};

  try {
    dmat = DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1,
                           cache_file, page_size);
  } catch (dmlc::Error& e) {
    std::vector<std::string> splited = common::Split(fname, '#');
    std::vector<std::string> args = common::Split(splited.front(), '?');
    std::string format {file_format};
    if (args.size() == 1 && file_format == "auto") {
      auto extension = common::Split(args.front(), '.').back();
      if (extension == "csv" || extension == "libsvm") {
        format = extension;
      }
      if (format == extension) {
        LOG(WARNING)
            << "No format parameter is provided in input uri, but found file extension: "
            << format << " .  "
            << "Consider providing a uri parameter: filename?format=" << format;
      } else {
        LOG(WARNING)
            << "No format parameter is provided in input uri.  "
            << "Choosing default parser in dmlc-core.  "
            << "Consider providing a uri parameter like: filename?format=csv";
      }
    }
    LOG(FATAL) << "Encountered parser error:\n" << e.what();
  }

  if (!silent) {
    LOG(CONSOLE) << dmat->Info().num_row_ << 'x' << dmat->Info().num_col_ << " matrix with "
                 << dmat->Info().num_nonzero_ << " entries loaded from " << uri;
  }
  /* sync up number of features after matrix loaded.
   * partitioned data will fail the train/val validation check
   * since partitioned data not knowing the real number of features. */
  rabit::Allreduce<rabit::op::Max>(&dmat->Info().num_col_, 1, nullptr,
    nullptr, fname.c_str());
  // backward compatiblity code.
  if (!load_row_split) {
    MetaInfo& info = dmat->Info();
    if (MetaTryLoadGroup(fname + ".group", &info.group_ptr_) && !silent) {
      LOG(CONSOLE) << info.group_ptr_.size() - 1
                   << " groups are loaded from " << fname << ".group";
    }
    if (MetaTryLoadFloatInfo
        (fname + ".base_margin", &info.base_margin_.HostVector()) && !silent) {
      LOG(CONSOLE) << info.base_margin_.Size()
                   << " base_margin are loaded from " << fname << ".base_margin";
    }
    if (MetaTryLoadFloatInfo
        (fname + ".weight", &info.weights_.HostVector()) && !silent) {
      LOG(CONSOLE) << info.weights_.Size()
                   << " weights are loaded from " << fname << ".weight";
    }
  }
  return dmat;
}

template <typename AdapterT>
DMatrix* DMatrix::Create(AdapterT* adapter, float missing, int nthread,
                         const std::string& cache_prefix,  size_t page_size) {
  if (cache_prefix.length() == 0) {
    // Data split mode is fixed to be row right now.
    return new data::SimpleDMatrix(adapter, missing, nthread);
  } else {
#if DMLC_ENABLE_STD_THREAD
    return new data::SparsePageDMatrix(adapter, missing, nthread, cache_prefix,
                                       page_size);
#else
    LOG(FATAL) << "External memory is not enabled in mingw";
    return nullptr;
#endif  // DMLC_ENABLE_STD_THREAD
  }
}

template DMatrix* DMatrix::Create<data::DenseAdapter>(
    data::DenseAdapter* adapter, float missing, int nthread,
    const std::string& cache_prefix, size_t page_size);
template DMatrix* DMatrix::Create<data::CSRAdapter>(
    data::CSRAdapter* adapter, float missing, int nthread,
    const std::string& cache_prefix, size_t page_size);
template DMatrix* DMatrix::Create<data::CSCAdapter>(
    data::CSCAdapter* adapter, float missing, int nthread,
    const std::string& cache_prefix, size_t page_size);
template DMatrix* DMatrix::Create<data::DataTableAdapter>(
    data::DataTableAdapter* adapter, float missing, int nthread,
    const std::string& cache_prefix, size_t page_size);
template DMatrix* DMatrix::Create<data::FileAdapter>(
    data::FileAdapter* adapter, float missing, int nthread,
    const std::string& cache_prefix, size_t page_size);
template DMatrix *
DMatrix::Create(data::IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext,
                                      XGBoostBatchCSR> *adapter,
                float missing, int nthread, const std::string &cache_prefix,
                size_t page_size);

SparsePage SparsePage::GetTranspose(int num_columns) const {
  SparsePage transpose;
  common::ParallelGroupBuilder<Entry, bst_row_t> builder(&transpose.offset.HostVector(),
                                                         &transpose.data.HostVector());
  const int nthread = omp_get_max_threads();
  builder.InitBudget(num_columns, nthread);
  long batch_size = static_cast<long>(this->Size());  // NOLINT(*)
#pragma omp parallel for default(none) shared(batch_size, builder) schedule(static)
  for (long i = 0; i < batch_size; ++i) {  // NOLINT(*)
    int tid = omp_get_thread_num();
    auto inst = (*this)[i];
    for (const auto& entry : inst) {
      builder.AddBudget(entry.index, tid);
    }
  }
  builder.InitStorage();
#pragma omp parallel for default(none) shared(batch_size, builder) schedule(static)
  for (long i = 0; i < batch_size; ++i) {  // NOLINT(*)
    int tid = omp_get_thread_num();
    auto inst = (*this)[i];
    for (const auto& entry : inst) {
      builder.Push(
          entry.index,
          Entry(static_cast<bst_uint>(this->base_rowid + i), entry.fvalue),
          tid);
    }
  }
  return transpose;
}
void SparsePage::Push(const SparsePage &batch) {
  auto& data_vec = data.HostVector();
  auto& offset_vec = offset.HostVector();
  const auto& batch_offset_vec = batch.offset.HostVector();
  const auto& batch_data_vec = batch.data.HostVector();
  size_t top = offset_vec.back();
  data_vec.resize(top + batch.data.Size());
  std::memcpy(dmlc::BeginPtr(data_vec) + top,
              dmlc::BeginPtr(batch_data_vec),
              sizeof(Entry) * batch.data.Size());
  size_t begin = offset.Size();
  offset_vec.resize(begin + batch.Size());
  for (size_t i = 0; i < batch.Size(); ++i) {
    offset_vec[i + begin] = top + batch_offset_vec[i + 1];
  }
}

template <typename AdapterBatchT>
uint64_t SparsePage::Push(const AdapterBatchT& batch, float missing, int nthread) {
  // Set number of threads but keep old value so we can reset it after
  const int nthreadmax = omp_get_max_threads();
  if (nthread <= 0) nthread = nthreadmax;
  int nthread_original = omp_get_max_threads();
  omp_set_num_threads(nthread);
  auto& offset_vec = offset.HostVector();
  auto& data_vec = data.HostVector();
  size_t builder_base_row_offset = this->Size();
  common::ParallelGroupBuilder<
      Entry, std::remove_reference<decltype(offset_vec)>::type::value_type>
      builder(&offset_vec, &data_vec, builder_base_row_offset);
  // Estimate expected number of rows by using last element in batch
  // This is not required to be exact but prevents unnecessary resizing
  size_t expected_rows = 0;
  if (batch.Size() > 0) {
    auto last_line = batch.GetLine(batch.Size() - 1);
    if (last_line.Size() > 0) {
      expected_rows =
          last_line.GetElement(last_line.Size() - 1).row_idx - base_rowid;
    }
  }
  builder.InitBudget(expected_rows, nthread);
  uint64_t max_columns = 0;

  // First-pass over the batch counting valid elements
  size_t num_lines = batch.Size();
#pragma omp parallel for schedule(static)
  for (omp_ulong i = 0; i < static_cast<omp_ulong>(num_lines);
       ++i) {  // NOLINT(*)
    int tid = omp_get_thread_num();
    auto line = batch.GetLine(i);
    for (auto j = 0ull; j < line.Size(); j++) {
      data::COOTuple element = line.GetElement(j);
      max_columns =
          std::max(max_columns, static_cast<uint64_t>(element.column_idx + 1));
      if (!common::CheckNAN(element.value) && element.value != missing) {
        size_t key = element.row_idx - base_rowid;
        // Adapter row index is absolute, here we want it relative to
        // current page
        CHECK_GE(key,  builder_base_row_offset);
        builder.AddBudget(key, tid);
      }
    }
  }
  builder.InitStorage();

  // Second pass over batch, placing elements in correct position
#pragma omp parallel for schedule(static)
  for (omp_ulong i = 0; i < static_cast<omp_ulong>(num_lines);
       ++i) {  // NOLINT(*)
    int tid = omp_get_thread_num();
    auto line = batch.GetLine(i);
    for (auto j = 0ull; j < line.Size(); j++) {
      auto element = line.GetElement(j);
      if (!common::CheckNAN(element.value) && element.value != missing) {
        size_t key = element.row_idx -
                     base_rowid;  // Adapter row index is absolute, here we want
                                  // it relative to current page
        builder.Push(key, Entry(element.column_idx, element.value), tid);
      }
    }
  }
  omp_set_num_threads(nthread_original);
  return max_columns;
}

void SparsePage::PushCSC(const SparsePage &batch) {
  std::vector<xgboost::Entry>& self_data = data.HostVector();
  std::vector<bst_row_t>& self_offset = offset.HostVector();

  auto const& other_data = batch.data.ConstHostVector();
  auto const& other_offset = batch.offset.ConstHostVector();

  if (other_data.empty()) {
    return;
  }
  if (!self_data.empty()) {
    CHECK_EQ(self_offset.size(), other_offset.size())
        << "self_data.size(): " << this->data.Size() << ", "
        << "other_data.size(): " << other_data.size() << std::flush;
  } else {
    self_data = other_data;
    self_offset = other_offset;
    return;
  }

  std::vector<bst_row_t> offset(other_offset.size());
  offset[0] = 0;

  std::vector<xgboost::Entry> data(self_data.size() + other_data.size());

  // n_cols in original csr data matrix, here in csc is n_rows
  size_t const n_features = other_offset.size() - 1;
  size_t beg = 0;
  size_t ptr = 1;
  for (size_t i = 0; i < n_features; ++i) {
    size_t const self_beg = self_offset.at(i);
    size_t const self_length = self_offset.at(i+1) - self_beg;
    // It is possible that the current feature and further features aren't referenced
    // in any rows accumulated thus far. It is also possible for this to happen
    // in the current sparse page row batch as well.
    // Hence, the incremental number of rows may stay constant thus equaling the data size
    CHECK_LE(beg, data.size());
    std::memcpy(dmlc::BeginPtr(data)+beg,
                dmlc::BeginPtr(self_data) + self_beg,
                sizeof(Entry) * self_length);
    beg += self_length;

    size_t const other_beg = other_offset.at(i);
    size_t const other_length = other_offset.at(i+1) - other_beg;
    CHECK_LE(beg, data.size());
    std::memcpy(dmlc::BeginPtr(data)+beg,
                dmlc::BeginPtr(other_data) + other_beg,
                sizeof(Entry) * other_length);
    beg += other_length;

    CHECK_LT(ptr, offset.size());
    offset.at(ptr) = beg;
    ptr++;
  }

  self_data = std::move(data);
  self_offset = std::move(offset);
}

namespace data {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(sparse_page_raw_format);
}  // namespace data
}  // namespace xgboost
