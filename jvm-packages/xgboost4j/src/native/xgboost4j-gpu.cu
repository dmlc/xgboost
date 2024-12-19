/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#include <jni.h>
#include <xgboost/c_api.h>

#include "../../../../src/common/cuda_pinned_allocator.h"
#include "../../../../src/common/device_vector.cuh"  // for device_vector
#include "../../../../src/data/array_interface.h"
#include "../../../../src/c_api/c_api_error.h"
#include "jvm_utils.h"
#include <fstream>


namespace xgboost {
namespace jni {

template <typename T, typename Alloc>
T const *RawPtr(std::vector<T, Alloc> const &data) {
  return data.data();
}

template <typename T, typename Alloc> T *RawPtr(std::vector<T, Alloc> &data) {
  return data.data();
}

template <typename T> T const *RawPtr(dh::device_vector<T> const &data) {
  return data.data().get();
}

template <typename T> T *RawPtr(dh::device_vector<T> &data) {
  return data.data().get();
}

template <typename T> T CheckJvmCall(T const &v, JNIEnv *jenv) {
  if (!v) {
    CHECK(jenv->ExceptionOccurred());
    jenv->ExceptionDescribe();
  }
  return v;
}

template <typename VCont>
void CopyColumnMask(xgboost::ArrayInterface<1> const &interface,
                    std::vector<Json> const &columns, cudaMemcpyKind kind,
                    size_t c, VCont *p_mask, Json *p_out, cudaStream_t stream) {
  auto &mask = *p_mask;
  auto &out = *p_out;
  auto size = sizeof(typename VCont::value_type) * interface.n;
  mask.resize(size);
  CHECK(RawPtr(mask));
  CHECK(size);
  CHECK(interface.valid.Data());
  dh::safe_cuda(
      cudaMemcpyAsync(RawPtr(mask), interface.valid.Data(), size, kind, stream));
  auto const &mask_column = columns[c]["mask"];
  out["mask"] = Object();
  std::vector<Json> mask_data{
      Json{reinterpret_cast<Integer::Int>(RawPtr(mask))},
      Json{get<Boolean const>(mask_column["data"][1])}};
  out["mask"]["data"] = Array(std::move(mask_data));
  if (get<Array const>(mask_column["shape"]).size() == 2) {
    std::vector<Json> mask_shape{
        Json{get<Integer const>(mask_column["shape"][0])},
        Json{get<Integer const>(mask_column["shape"][1])}};
    out["mask"]["shape"] = Array(std::move(mask_shape));
  } else if (get<Array const>(mask_column["shape"]).size() == 1) {
    std::vector<Json> mask_shape{
        Json{get<Integer const>(mask_column["shape"][0])}};
    out["mask"]["shape"] = Array(std::move(mask_shape));
  } else {
    LOG(FATAL) << "Invalid shape of mask";
  }
  out["mask"]["typestr"] = String("<t1");
  out["mask"]["version"] = Integer(3);
}

template <typename DCont, typename VCont>
void CopyInterface(std::vector<xgboost::ArrayInterface<1>> &interface_arr,
                   std::vector<Json> const &columns, cudaMemcpyKind kind,
                   std::vector<DCont> *p_data, std::vector<VCont> *p_mask,
                   std::vector<xgboost::Json> *p_out, cudaStream_t stream) {
  p_data->resize(interface_arr.size());
  p_mask->resize(interface_arr.size());
  p_out->resize(interface_arr.size());
  for (size_t c = 0; c < interface_arr.size(); ++c) {
    auto &interface = interface_arr.at(c);
    size_t element_size = interface.ElementSize();
    size_t size = element_size * interface.n;

    auto &data = (*p_data)[c];
    auto &mask = (*p_mask)[c];
    data.resize(size);
    dh::safe_cuda(cudaMemcpyAsync(RawPtr(data), interface.data, size, kind, stream));

    auto &out = (*p_out)[c];
    out = Object();
    std::vector<Json> j_data{
        Json{Integer(reinterpret_cast<Integer::Int>(RawPtr(data)))},
        Json{Boolean{false}}};

    out["data"] = Array(std::move(j_data));
    out["shape"] = Array(std::vector<Json>{Json(Integer(interface.Shape<0>()))});

    if (interface.valid.Data()) {
      CopyColumnMask(interface, columns, kind, c, &mask, &out, stream);
    }
    out["typestr"] = String("<f4");
    out["version"] = Integer(3);
  }
}

template <typename T>
void CopyMetaInfo(Json *p_interface, dh::device_vector<T> *out, cudaStream_t stream) {
  auto &j_interface = *p_interface;
  CHECK_EQ(get<Array const>(j_interface).size(), 1);
  auto object = get<Object>(get<Array>(j_interface)[0]);
  ArrayInterface<1> interface(object);
  out->resize(interface.Shape<0>());
  size_t element_size = interface.ElementSize();
  size_t size = element_size * interface.n;
  dh::safe_cuda(cudaMemcpyAsync(RawPtr(*out), interface.data, size,
                                cudaMemcpyDeviceToDevice, stream));
  j_interface[0]["data"][0] = reinterpret_cast<Integer::Int>(RawPtr(*out));
}

template <typename DCont, typename VCont> struct DataFrame {
  std::vector<DCont> data;
  std::vector<VCont> valid;
  std::vector<Json> interfaces;
};

// The base class for external memory
class ExternalMemory {
 public:
  // Load data from the exact external memory to the GPU
  virtual void LoadData(size_t batch_number,
                        std::vector<xgboost::Json> *p_out,
                        cudaStream_t stream) = 0;

  // Stage data into the exact external memory from GPU.
  virtual void StageData(
      std::vector<xgboost::ArrayInterface<1>> &interface_arr,
      std::vector<Json> const &columns,
      cudaStream_t stream) = 0;

  virtual ~ExternalMemory() = default;

 protected:
  // Temp buffer on device, each `dh::device_vector` represents a column from cudf.
  std::vector<dh::device_vector<char>> staging_data_;
  std::vector<dh::device_vector<uint8_t>> staging_mask_;
};

// The data will be stored on CPU memory
class HostExternalMemory: public ExternalMemory {
 public:
  void StageData(std::vector<xgboost::ArrayInterface<1>> &interfaces,
                 std::vector<Json> const &columns,
                 cudaStream_t stream) override {
    std::cerr << "HostExternalMemory StageData batch number: " << std::endl;
    // DataFrame
    using T = decltype(host_columns_)::value_type::element_type;
    host_columns_.emplace_back(std::unique_ptr<T>(new T));

    CopyInterface(interfaces, columns, cudaMemcpyDeviceToHost, &host_columns_.back()->data,
                  &host_columns_.back()->valid, &host_columns_.back()->interfaces, stream);
  }

  void LoadData(size_t batch_number,
                std::vector<xgboost::Json> *p_out,
                cudaStream_t stream) override {
    std::cerr << "HostExternalMemory LoadData batch number: " << batch_number << std::endl;
    // Data
    auto const &json_interface = host_columns_.at(batch_number)->interfaces;
    std::vector<ArrayInterface<1>> in;
    for (auto interface : json_interface) {
      auto column = ArrayInterface<1>(get<Object const>(interface));
      in.emplace_back(column);
    }
    CopyInterface(in, json_interface, cudaMemcpyHostToDevice, &staging_data_,
                  &staging_mask_, p_out, nullptr);
  }

 private:
    template <typename T>
    using Alloc = xgboost::common::cuda_impl::PinnedAllocator<T>;
    template <typename U>
    using HostVector = std::vector<U, Alloc<U>>;
    // This vector is created for staging device data on host to save GPU memory.
    // When space is not of concern, we can stage them on device memory directly.
    std::vector<
        std::unique_ptr<DataFrame<HostVector<char>, HostVector<std::uint8_t>>>>
        host_columns_;
};

struct DataFile {
  std::string path;
  size_t size;
  size_t shape;
};

struct FilesDataFrame {
  std::vector<DataFile> data_files;
  std::vector<DataFile> valid_files;
};

// The data will be cached into local disk.
class DiskExternalMemory: public ExternalMemory {
 public:
  DiskExternalMemory(std::string root): root_(root) {}

  void LoadData(size_t batch_number, std::vector<xgboost::Json> *p_out,
                cudaStream_t stream) override {
    std::cerr << " DiskExternalMemory LoadData " << std::endl;
    auto files_data_frame = staged_files_.at(batch_number);
    auto col_number = files_data_frame.data_files.size();

    p_out->resize(col_number);
    staging_data_.resize(col_number);

    for (size_t c = 0; c < col_number; c++) {
      auto data_file = files_data_frame.data_files.at(c);

      std::vector<char> host_data;
      host_data.resize(data_file.size);
      std::ifstream inputFile(data_file.path, std::ios::binary);
      if (inputFile.is_open()) {
        inputFile.read(RawPtr(host_data), data_file.size);
        inputFile.close();
      } else {
        std::cerr << "Failed to open file for reading" << std::endl;
      }

      auto device_data = staging_data_.at(c);
      device_data.resize(data_file.size);
      dh::safe_cuda(cudaMemcpyAsync(RawPtr(device_data), RawPtr(host_data), data_file.size,
                                    cudaMemcpyHostToDevice, stream));
      auto &out = (*p_out).at(c);
      out = Object();
      std::vector<Json> j_data{Json{Integer(reinterpret_cast<Integer::Int>(RawPtr(device_data)))},
                               Json{Boolean{false}}};

      out["data"] = Array(std::move(j_data));
      out["shape"] = Array(std::vector<Json>{Json(Integer(data_file.shape))});
      out["typestr"] = String("<f4");
      out["version"] = Integer(3);
      // TODO support mask
    }
  }

  void StageData(std::vector<xgboost::ArrayInterface<1>> &interface_arr,
                 const std::vector<Json> &columns, cudaStream_t stream) override {
    std::cerr << " DiskExternalMemory StageData " << std::endl;
    ++n_batches_;
    FilesDataFrame files_data_frame;
    for (size_t c = 0; c < interface_arr.size(); ++c) {
      auto &interface = interface_arr.at(c);
      size_t element_size = interface.ElementSize();
      size_t size = element_size * interface.n;

      std::vector<char> tmp;
      tmp.resize(size);
      dh::safe_cuda(cudaMemcpyAsync(RawPtr(tmp), interface.data, size,
                                    cudaMemcpyDeviceToHost, stream));

      std::string file_name = GenerateFileName(n_batches_, c, "data");
      WriteDataToFile(file_name, RawPtr(tmp), size);

      // Store the information.
      files_data_frame.data_files.emplace_back(DataFile{file_name, size, interface.Shape<0>()});

      if (interface.valid.Data()) {
        // TODO support mask
      }
    }
    staged_files_.emplace_back(files_data_frame);
  }

  ~DiskExternalMemory() override {
    for (auto cached_files: staged_files_) {
        for (auto data_file : cached_files.data_files) {
            std::cerr << "removing " << data_file.path.c_str() << std::endl;
            std::remove(data_file.path.c_str());
        }
        for (auto data_file : cached_files.valid_files) {
            std::remove(data_file.path.c_str());
        }
    }
  };

 private:
  std::string GenerateFileName(size_t batch_number, size_t column, std::string type) const {
      std::stringstream ss;
      ss << root_ << "/" << batch_number << "_" << column << "_" << type << ".bin";
      return ss.str();
  }

  void WriteDataToFile(std::string path, char *data, size_t size) const {
    std::ofstream output_file(path, std::ios::binary);
    if (output_file.is_open()) {
      output_file.write(data, size);
      output_file.close();
    } else {
      std::cerr << "Failed to open file " << std::endl;
    }
  }

  std::vector<FilesDataFrame> staged_files_;
  const std::string root_; // the root path of external memory
  size_t n_batches_ = 0;
};

class DataIteratorProxy {
  DMatrixHandle proxy_;
  JNIEnv *jenv_;
  int jni_status_;
  jobject jiter_;
  std::unique_ptr<ExternalMemory> ext_memory_;

  // Staging area for metainfo.
  // TODO(Bobby): label_upper_bound, label_lower_bound, group.
  std::vector<std::unique_ptr<dh::device_vector<float>>> labels_;
  std::vector<std::unique_ptr<dh::device_vector<float>>> weights_;
  std::vector<std::unique_ptr<dh::device_vector<float>>> base_margins_;
  std::vector<std::unique_ptr<dh::device_vector<int>>> qids_;
  std::vector<Json> label_interfaces_;
  std::vector<Json> weight_interfaces_;
  std::vector<Json> margin_interfaces_;
  std::vector<Json> qid_interfaces_;

  size_t it_{0};
  size_t n_batches_{0};
  bool initialized_{false};
  jobject last_batch_ {nullptr};

  cudaStream_t copy_stream_;

 public:
  explicit DataIteratorProxy(jobject jiter, std::string external_path): jiter_{jiter} {
    if (!external_path.empty()) {
      ext_memory_ = std::make_unique<DiskExternalMemory>(external_path);
    } else {
      ext_memory_ = std::make_unique<HostExternalMemory>();
    }
    XGProxyDMatrixCreate(&proxy_);
    jni_status_ =
        GlobalJvm()->GetEnv(reinterpret_cast<void **>(&jenv_), JNI_VERSION_1_6);
    this->Reset();
    dh::safe_cuda(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
  }
  ~DataIteratorProxy() {
    XGDMatrixFree(proxy_);
    dh::safe_cuda(cudaStreamDestroy(copy_stream_));
  }

  DMatrixHandle GetDMatrixHandle() const { return proxy_; }

  // Helper function for staging meta info.
  void StageMetaInfo(Json json_interface) {
    CHECK(!IsA<Null>(json_interface));
    auto json_map = get<Object const>(json_interface);
    if (json_map.find("label") == json_map.cend()) {
      LOG(FATAL) << "Must have a label field.";
    }

    Json label = json_interface["label"];
    CHECK(!IsA<Null>(label));
    labels_.emplace_back(new dh::device_vector<float>);
    CopyMetaInfo(&label, labels_.back().get(), copy_stream_);
    label_interfaces_.emplace_back(label);

    std::string str;
    Json::Dump(label, &str);
    XGDMatrixSetInfoFromInterface(proxy_, "label", str.c_str());

    if (json_map.find("weight") != json_map.cend()) {
      Json weight = json_interface["weight"];
      CHECK(!IsA<Null>(weight));
      weights_.emplace_back(new dh::device_vector<float>);
      CopyMetaInfo(&weight, weights_.back().get(), copy_stream_);
      weight_interfaces_.emplace_back(weight);

      Json::Dump(weight, &str);
      XGDMatrixSetInfoFromInterface(proxy_, "weight", str.c_str());
    }

    if (json_map.find("baseMargin") != json_map.cend()) {
      Json basemargin = json_interface["baseMargin"];
      base_margins_.emplace_back(new dh::device_vector<float>);
      CopyMetaInfo(&basemargin, base_margins_.back().get(), copy_stream_);
      margin_interfaces_.emplace_back(basemargin);

      Json::Dump(basemargin, &str);
      XGDMatrixSetInfoFromInterface(proxy_, "base_margin", str.c_str());
    }

    if (json_map.find("qid") != json_map.cend()) {
      Json qid = json_interface["qid"];
      qids_.emplace_back(new dh::device_vector<int>);
      CopyMetaInfo(&qid, qids_.back().get(), copy_stream_);
      qid_interfaces_.emplace_back(qid);

      Json::Dump(qid, &str);
      XGDMatrixSetInfoFromInterface(proxy_, "qid", str.c_str());
    }
  }

  void CloseJvmBatch() {
    if (last_batch_) {
      jclass batch_class = CheckJvmCall(jenv_->GetObjectClass(last_batch_), jenv_);
      jmethodID closeMethod = CheckJvmCall(jenv_->GetMethodID(batch_class, "close", "()V"), jenv_);
      jenv_->CallVoidMethod(last_batch_, closeMethod);
      last_batch_ = nullptr;
    }
  }

  void Reset() {
    it_ = 0;
    this->CloseJvmBatch();
  }

  int32_t PullIterFromJVM() {
    jclass iterClass = jenv_->FindClass("java/util/Iterator");
    this->CloseJvmBatch();

    jmethodID has_next =
        CheckJvmCall(jenv_->GetMethodID(iterClass, "hasNext", "()Z"), jenv_);
    jmethodID next = CheckJvmCall(
        jenv_->GetMethodID(iterClass, "next", "()Ljava/lang/Object;"), jenv_);

    if (jenv_->CallBooleanMethod(jiter_, has_next)) {
      // batch should be ColumnBatch from jvm
      jobject batch = CheckJvmCall(jenv_->CallObjectMethod(jiter_, next), jenv_);
      jclass batch_class = CheckJvmCall(jenv_->GetObjectClass(batch), jenv_);
      jmethodID toJson = CheckJvmCall(jenv_->GetMethodID(
        batch_class, "toJson", "()Ljava/lang/String;"), jenv_);

      auto jinterface =
        static_cast<jstring>(jenv_->CallObjectMethod(batch, toJson));
      CheckJvmCall(jinterface, jenv_);
      char const *c_interface_str =
          CheckJvmCall(jenv_->GetStringUTFChars(jinterface, nullptr), jenv_);

      StageData(c_interface_str);

      jenv_->ReleaseStringUTFChars(jinterface, c_interface_str);

      last_batch_ = batch;
      return 1;
    } else {
      return 0;
    }
  }

  void StageData(std::string interface_str) {
    ++n_batches_;

    // Stage the meta info.
    auto json_interface =
        Json::Load({interface_str.c_str(), interface_str.size()});
    CHECK(!IsA<Null>(json_interface));
    StageMetaInfo(json_interface);

    Json features = json_interface["features"];
    auto json_columns = get<Array const>(features);
    std::vector<ArrayInterface<1>> interfaces;

    // Stage the data
    for (auto &json_col : json_columns) {
      auto column = ArrayInterface<1>(get<Object const>(json_col));
      interfaces.emplace_back(column);
    }
    ext_memory_->StageData(interfaces, json_columns, copy_stream_);

    Json::Dump(features, &interface_str);
    XGProxyDMatrixSetDataCudaColumnar(proxy_, interface_str.c_str());
    it_++;
  }

  int NextFirstLoop() {
    try {
      dh::safe_cuda(cudaStreamSynchronize(copy_stream_));
      if (this->PullIterFromJVM()) {
        return 1;
      } else {
        initialized_ = true;
        return 0;
      }
    } catch (dmlc::Error const &e) {
      if (jni_status_ == JNI_EDETACHED) {
        GlobalJvm()->DetachCurrentThread();
      }
      LOG(FATAL) << e.what();
    }
    LOG(FATAL) << "Unreachable";
    return 1;
  }

  int NextSecondLoop() {
    std::string str;
    // Meta
    auto const &label = this->label_interfaces_.at(it_);
    Json::Dump(label, &str);
    XGDMatrixSetInfoFromInterface(proxy_, "label", str.c_str());

    if (n_batches_ == this->weight_interfaces_.size()) {
      auto const &weight = this->weight_interfaces_.at(it_);
      Json::Dump(weight, &str);
      XGDMatrixSetInfoFromInterface(proxy_, "weight", str.c_str());
    }

    if (n_batches_ == this->margin_interfaces_.size()) {
      auto const &base_margin = this->margin_interfaces_.at(it_);
      Json::Dump(base_margin, &str);
      XGDMatrixSetInfoFromInterface(proxy_, "base_margin", str.c_str());
    }

    if (n_batches_ == this->qid_interfaces_.size()) {
      auto const &qid = this->qid_interfaces_.at(it_);
      Json::Dump(qid, &str);
      XGDMatrixSetInfoFromInterface(proxy_, "qid", str.c_str());
    }

    std::vector<Json> out;
    ext_memory_->LoadData(it_, &out, copy_stream_);

    Json temp{Array(std::move(out))};
    std::string interface_str;
    Json::Dump(temp, &interface_str);
    std::cerr << "NextSecondLoop " << interface_str << std::endl;
    XGProxyDMatrixSetDataCudaColumnar(proxy_, interface_str.c_str());
    it_++;
    return 1;
  }

  int Next() {
    if (!initialized_) {
      return NextFirstLoop();
    } else {
      if (it_ == n_batches_) {
        return 0;
      }
      return NextSecondLoop();
    }
  };
};

namespace {
void Reset(DataIterHandle self) {
  static_cast<xgboost::jni::DataIteratorProxy *>(self)->Reset();
}

int Next(DataIterHandle self) {
  return static_cast<xgboost::jni::DataIteratorProxy *>(self)->Next();
}

template <typename T>
using Deleter = std::function<void(T *)>;
} // anonymous namespace

XGB_DLL int XGQuantileDMatrixCreateFromCallbackImpl(JNIEnv *jenv, jclass, jobject jdata_iter,
                                                    jlongArray jref, char const *config,
                                                    jlongArray jout) {
  xgboost_CHECK_C_ARG_PTR(config);
  auto jconfig = Json::Load(StringView{config});
  auto ext_mem_path = OptionalArg<String>(jconfig, "external_memory_path", std::string(""));
  std::cerr << "XGQuantileDMatrixCreateFromCallbackImpl external_memory_path " << ext_mem_path << std::endl;

  xgboost::jni::DataIteratorProxy proxy(jdata_iter, ext_mem_path);
  DMatrixHandle result;
  DMatrixHandle ref{nullptr};

  if (jref != nullptr) {
    std::unique_ptr<jlong, Deleter<jlong>> refptr{jenv->GetLongArrayElements(jref, nullptr),
                                                  [&](jlong *ptr) {
                                                    jenv->ReleaseLongArrayElements(jref, ptr, 0);
                                                    jenv->DeleteLocalRef(jref);
                                                  }};
    ref = reinterpret_cast<DMatrixHandle>(refptr.get()[0]);
  }

  auto ret = XGQuantileDMatrixCreateFromCallback(&proxy, proxy.GetDMatrixHandle(), ref, Reset, Next,
                                                 config, &result);
  setHandle(jenv, jout, result);
  return ret;
}
} // namespace jni
} // namespace xgboost
