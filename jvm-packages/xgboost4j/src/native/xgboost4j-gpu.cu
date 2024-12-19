/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#include <jni.h>
#include <xgboost/c_api.h>

#include <filesystem>

#include "../../../../src/c_api/c_api_error.h"       // for XGBAPIHandleException
#include "../../../../src/common/device_vector.cuh"  // for device_vector
#include "../../../../src/common/io.h"
#include "../../../../src/data/array_interface.h"
#include "../../../../src/data/sparse_page_source.h"  // for Cache
#include "jvm_utils.h"

namespace xgboost::jni {
template <typename T, typename Alloc>
T const *RawPtr(std::vector<T, Alloc> const &data) {
  return data.data();
}

template <typename T, typename Alloc>
T *RawPtr(std::vector<T, Alloc> &data) {
  return data.data();
}

template <typename T>
T const *RawPtr(dh::device_vector<T> const &data) {
  return data.data().get();
}

template <typename T>
T *RawPtr(dh::device_vector<T> &data) {
  return data.data().get();
}

template <typename T>
T CheckJvmCall(T const &v, JNIEnv *jenv) {
  if (!v) {
    CHECK(jenv->ExceptionOccurred());
    jenv->ExceptionDescribe();
  }
  return v;
}

void CheckXgbCall(JNIEnv *, std::int32_t ret) {
  if (ret != 0) {
    auto const *msg = XGBGetLastError();
    LOG(FATAL) << msg;
  }
}

template <typename T>
void CopyMetaInfo(Json *p_interface, dh::device_vector<T> *out) {
  auto &j_interface = *p_interface;
  CHECK_EQ(get<Array const>(j_interface).size(), 1);
  auto object = get<Object>(get<Array>(j_interface)[0]);
  ArrayInterface<1> interface(object);
  out->resize(interface.Shape<0>());
  size_t element_size = interface.ElementSize();
  size_t size = element_size * interface.n;
  dh::safe_cuda(cudaMemcpyAsync(RawPtr(*out), interface.data, size, cudaMemcpyDeviceToDevice));
  j_interface[0]["data"][0] = reinterpret_cast<Integer::Int>(RawPtr(*out));
}

template <typename DCont, typename VCont>
struct DataFrame {
  std::vector<DCont> data;
  std::vector<VCont> valid;
  std::vector<Json> interfaces;
  void Clear() {
    data.clear();
    valid.clear();
    interfaces.clear();
  }
};

namespace fs = std::filesystem;

using ColumnBuf = std::unique_ptr<HostDeviceVector<std::int8_t>>;
using ColumnMaskBuf = std::unique_ptr<HostDeviceVector<std::int8_t>>;

class JvmCacheFormat {
  data::Cache cache_info_;
  std::size_t n_columns_{0};
  DeviceOrd device_;

  struct ColumnInfo {
    bool masked{false};
    std::size_t n_samples{0};
    char typechar{0};
  };
  std::vector<std::vector<ColumnInfo>> column_info_;

 public:
  explicit JvmCacheFormat() : cache_info_{false, "jvm-cache", "raw", false} {}

  /**
   * @brief Write a device dataframe to the disk.
   */
  std::size_t Write(common::AlignedWriteStream *fo, std::vector<ArrayInterface<1>> const &df) {
    std::uint64_t n_total_bytes{0};
    std::size_t n_columns{0};
    std::vector<ColumnInfo> info;
    for (auto const &col : df) {
      std::uint64_t n_bytes = col.ElementSize() * col.n;
      std::vector<std::byte> h_bytes(n_bytes);
      if (device_.IsCPU()) {
        device_ = DeviceOrd::CUDA(dh::CudaGetPointerDevice(col.data));
      }
      std::cerr << "write: n_bytes" << n_bytes << std::endl;
      dh::safe_cuda(cudaMemcpy(h_bytes.data(), col.data, n_bytes, cudaMemcpyDefault));

      auto written = fo->Write(&n_bytes, sizeof(n_bytes));  // write the size
      written += fo->Write(h_bytes.data(), n_bytes);        // write the data
      n_bytes = written;

      ColumnInfo cinfo;
      cinfo.n_samples = col.Shape<0>();
      DispatchDType(col.type, [&](auto t) {
        using T = decltype(t);
        auto c = linalg::detail::ArrayInterfaceHandler::TypeChar<T>();
        cinfo.typechar = c;
      });

      if (col.valid.Data()) {
        cinfo.masked = true;
        std::uint64_t n_mask_bytes = col.valid.Capacity() / sizeof(std::byte);
        std::vector<std::byte> h_mask_bytes(n_mask_bytes);
        dh::safe_cuda(
            cudaMemcpy(h_mask_bytes.data(), col.valid.Data(), n_mask_bytes, cudaMemcpyDefault));
        n_bytes += fo->Write(h_mask_bytes.data(), n_mask_bytes);
        n_bytes += fo->Write(&n_mask_bytes, sizeof(n_mask_bytes));
      }

      n_total_bytes += n_bytes;
      n_columns++;

      info.emplace_back(cinfo);
      std::cerr << "column:" << n_columns << std::endl;
    }

    if (this->n_columns_ == 0) {
      this->n_columns_ = n_columns;
    }
    CHECK_EQ(this->n_columns_, n_columns);
    this->column_info_.emplace_back(info);

    cache_info_.Push(n_total_bytes);
    return n_total_bytes;
  }

  void Commit() { this->cache_info_.Commit(); }

  /**
   * @brief Read a device dataframe from the stream.
   */
  void Read(std::int32_t iter, common::AlignedResourceReadStream *fi,
            DataFrame<ColumnBuf, ColumnMaskBuf> *p_out) const {
    p_out->Clear();
    auto info = this->cache_info_.View(iter);
    auto const &binfo = this->column_info_.at(iter);

    for (std::size_t i = 0; i < this->n_columns_; ++i) {
      std::uint64_t n_bytes{0};
      CHECK(fi->Read(&n_bytes));
      auto [ptr, size] = fi->Consume(n_bytes);
      CHECK_EQ(size, n_bytes);
      auto p_data = std::make_unique<ColumnBuf::element_type>();
      p_data->SetDevice(this->device_);
      p_data->Resize(n_bytes);
      dh::safe_cuda(cudaMemcpy(p_data->DevicePointer(), ptr, n_bytes, cudaMemcpyDefault));

      p_out->data.emplace_back(std::move(p_data));

      ColumnInfo const &cinfo = binfo.at(i);
      Json jcol{Object{}};
      jcol["data"] = Array{std::vector<Json>{
          Json{Integer{reinterpret_cast<Integer::Int>(p_out->data.back()->ConstDevicePointer())}},
          Json{Boolean{true}}}};
      jcol["shape"] =
          Array{std::vector<Json>{Json{Integer{static_cast<Integer::Int>(cinfo.n_samples)}}}};
      CHECK_EQ(n_bytes % cinfo.n_samples, 0);
      std::size_t elem_bytes = n_bytes / cinfo.n_samples;
      jcol["typestr"] = String{"<" + (cinfo.typechar + std::to_string(elem_bytes))};
      jcol["version"] = Integer{3};
      jcol["stream"] = Integer{2};

      if (cinfo.masked) {
        std::uint64_t n_mask_bytes{0};
        CHECK(fi->Read(&n_mask_bytes));
        auto p_mask = std::make_unique<ColumnMaskBuf::element_type>();
        p_mask->SetDevice(this->device_);
        p_mask->Resize(n_mask_bytes);
        auto [ptr, size] = fi->Consume(n_mask_bytes);
        CHECK_EQ(size, n_mask_bytes);
        dh::safe_cuda(cudaMemcpy(p_mask->DevicePointer(), ptr, n_mask_bytes, cudaMemcpyDefault));

        Json jmask{Object{}};

        std::vector<Json> mask_data{
            Json{reinterpret_cast<Integer::Int>(p_mask->ConstDevicePointer())}, Json{true}};
        jmask["data"] = Array(std::move(mask_data));
        jmask["shape"] =
            Array{std::vector<Json>{Json{Integer{static_cast<Integer::Int>(cinfo.n_samples)}}}};
        jmask["typestr"] = String{"<t1"};
        jmask["version"] = Integer{3};
        jmask["stream"] = Integer{2};
        jcol["mask"] = jmask;

        p_out->valid.emplace_back(std::move(p_mask));
      } else {
        p_out->valid.emplace_back(nullptr);
      }

      p_out->interfaces.emplace_back(jcol);
    }
  }

  auto View(std::int32_t iter) const { return this->cache_info_.View(iter); }
};

class DataIteratorProxy {
  DMatrixHandle proxy_;
  JNIEnv *jenv_;
  int jni_status_;
  jobject jiter_;
  bool cache_on_host_{true};  // TODO(Bobby): Make this optional.
  fs::path root_;

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
  jobject last_batch_{nullptr};

  // Temp buffer on device for set data calls.
  DataFrame<ColumnBuf, ColumnMaskBuf> out_df_;

  std::unique_ptr<JvmCacheFormat> cache_{std::make_unique<JvmCacheFormat>()};
  std::string host_buf_;
  std::size_t host_buf_offset_{0};  // for stream write offset

 public:
  explicit DataIteratorProxy(jobject jiter, fs::path root)
      : jiter_{jiter}, cache_on_host_{root.empty()}, root_{std::move(root)} {
    CHECK(fs::exists(root_)) << "Directory `" << root_.string() << "` not found.";
    CheckXgbCall(jenv_, XGProxyDMatrixCreate(&proxy_));
    jni_status_ = GlobalJvm()->GetEnv(reinterpret_cast<void **>(&jenv_), JNI_VERSION_1_6);
    this->Reset();
  }
  ~DataIteratorProxy() { CheckXgbCall(jenv_, XGDMatrixFree(proxy_)); }

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
    CopyMetaInfo(&label, labels_.back().get());
    label_interfaces_.emplace_back(label);

    std::string str;
    Json::Dump(label, &str);
    XGDMatrixSetInfoFromInterface(proxy_, "label", str.c_str());

    auto it = json_map.find("weight");
    if (it != json_map.cend()) {
      Json weight = it->second;
      CHECK(!IsA<Null>(weight));
      weights_.emplace_back(new dh::device_vector<float>);
      CopyMetaInfo(&weight, weights_.back().get());
      weight_interfaces_.emplace_back(weight);

      Json::Dump(weight, &str);
      XGDMatrixSetInfoFromInterface(proxy_, "weight", str.c_str());
    }

    it = json_map.find("baseMargin");
    if (it != json_map.cend()) {
      Json basemargin = it->second;
      base_margins_.emplace_back(new dh::device_vector<float>);
      CopyMetaInfo(&basemargin, base_margins_.back().get());
      margin_interfaces_.emplace_back(basemargin);

      Json::Dump(basemargin, &str);
      XGDMatrixSetInfoFromInterface(proxy_, "base_margin", str.c_str());
    }

    it = json_map.find("qid");
    if (it != json_map.cend()) {
      Json qid = it->second;
      qids_.emplace_back(new dh::device_vector<int>);
      CopyMetaInfo(&qid, qids_.back().get());
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

  bool PullIterFromJVM() {
    jclass iterClass = jenv_->FindClass("java/util/Iterator");
    this->CloseJvmBatch();

    jmethodID has_next = CheckJvmCall(jenv_->GetMethodID(iterClass, "hasNext", "()Z"), jenv_);
    jmethodID next =
        CheckJvmCall(jenv_->GetMethodID(iterClass, "next", "()Ljava/lang/Object;"), jenv_);

    if (jenv_->CallBooleanMethod(jiter_, has_next)) {
      // batch should be ColumnBatch from jvm
      jobject batch = CheckJvmCall(jenv_->CallObjectMethod(jiter_, next), jenv_);
      jclass batch_class = CheckJvmCall(jenv_->GetObjectClass(batch), jenv_);
      jmethodID toJson =
          CheckJvmCall(jenv_->GetMethodID(batch_class, "toJson", "()Ljava/lang/String;"), jenv_);

      auto jinterface = static_cast<jstring>(jenv_->CallObjectMethod(batch, toJson));
      CheckJvmCall(jinterface, jenv_);

      auto release = [&](char const *ptr) {
        if (ptr) {
          jenv_->ReleaseStringUTFChars(jinterface, ptr);
        }
      };
      std::unique_ptr<char const, decltype(release)> jinf{
          CheckJvmCall(jenv_->GetStringUTFChars(jinterface, nullptr), jenv_), release};

      this->StageData(jinf.get());

      last_batch_ = batch;
      return true;
    } else {
      return false;
    }
  }

  void StageData(std::string interface_str) {
    ++n_batches_;
    // Stage the meta info.
    auto json_interface = Json::Load(interface_str);
    CHECK(!IsA<Null>(json_interface));
    StageMetaInfo(json_interface);

    Json features = json_interface["features"];
    auto json_columns = get<Array const>(features);
    std::vector<ArrayInterface<1>> interfaces;

    // Stage the data
    for (auto &json_col : json_columns) {
      auto column = ArrayInterface<1>{get<Object const>(json_col)};
      interfaces.emplace_back(column);
    }
    interface_str = Json::Dump(features);  // reuse

    if (this->cache_on_host_) {
      auto fo = std::make_unique<common::AlignedMemWriteStream>(&host_buf_, host_buf_offset_);
      host_buf_offset_ += this->cache_->Write(fo.get(), interfaces);
    } else {
      auto path = root_ / "X";
      auto fo = std::make_unique<common::AlignedFileWriteStream>(path.string(), "ab");
      host_buf_offset_ += this->cache_->Write(fo.get(), interfaces);
    }
    CheckXgbCall(jenv_, XGProxyDMatrixSetDataCudaColumnar(proxy_, interface_str.c_str()));

    it_++;
  }

  int NextFirstLoop() {
    try {
      if (this->PullIterFromJVM()) {
        return 1;
      } else {
        initialized_ = true;
        this->cache_->Commit();
        return 0;
      }
    } catch (dmlc::Error const &e) {
      if (jni_status_ == JNI_EDETACHED) {
        GlobalJvm()->DetachCurrentThread();
      }
      XGBAPIHandleException(e);
      return 0;
    }
    LOG(FATAL) << "Unreachable";
    return 1;
  }

  int NextSecondLoop() {
    std::string str;
    // Meta
    auto const &label = this->label_interfaces_.at(it_);
    Json::Dump(label, &str);
    CheckXgbCall(jenv_, XGDMatrixSetInfoFromInterface(proxy_, "label", str.c_str()));

    if (n_batches_ == this->weight_interfaces_.size()) {
      auto const &weight = this->weight_interfaces_.at(it_);
      Json::Dump(weight, &str);
      CheckXgbCall(jenv_, XGDMatrixSetInfoFromInterface(proxy_, "weight", str.c_str()));
    }

    if (n_batches_ == this->margin_interfaces_.size()) {
      auto const &base_margin = this->margin_interfaces_.at(it_);
      Json::Dump(base_margin, &str);
      CheckXgbCall(jenv_, XGDMatrixSetInfoFromInterface(proxy_, "base_margin", str.c_str()));
    }

    if (n_batches_ == this->qid_interfaces_.size()) {
      auto const &qid = this->qid_interfaces_.at(it_);
      Json::Dump(qid, &str);
      CheckXgbCall(jenv_, XGDMatrixSetInfoFromInterface(proxy_, "qid", str.c_str()));
    }

    // Data
    auto info = this->cache_->View(this->it_);
    if (this->cache_on_host_) {
      auto res = std::make_shared<common::FixedBufferResource>(host_buf_, info.first);
      auto fi = std::make_unique<common::AlignedResourceReadStream>(res);
      this->cache_->Read(this->it_, fi.get(), &out_df_);
    } else {
      auto path = root_ / "X";
      auto fi =
          std::make_unique<common::PrivateMmapConstStream>(path.string(), info.first, info.second);
      this->cache_->Read(this->it_, fi.get(), &out_df_);
    }

    Json temp{Array{out_df_.interfaces}};
    std::string interface_str;
    Json::Dump(temp, &interface_str);
    CheckXgbCall(jenv_, XGProxyDMatrixSetDataCudaColumnar(proxy_, interface_str.c_str()));
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
      try {
        return NextSecondLoop();
      } catch (dmlc::Error const &e) {
        if (jni_status_ == JNI_EDETACHED) {
          GlobalJvm()->DetachCurrentThread();
        }
        XGBAPIHandleException(e);
        return 0;
      }
    }
  };
};

namespace {
void Reset(DataIterHandle self) { static_cast<xgboost::jni::DataIteratorProxy *>(self)->Reset(); }

int Next(DataIterHandle self) {
  return static_cast<xgboost::jni::DataIteratorProxy *>(self)->Next();
}

template <typename T>
using Deleter = std::function<void(T *)>;
}  // anonymous namespace

XGB_DLL int XGQuantileDMatrixCreateFromCallbackImpl(JNIEnv *jenv, jclass, jobject jdata_iter,
                                                    jlongArray jref, char const *config,
                                                    jlongArray jout) {
  xgboost_CHECK_C_ARG_PTR(config);
  auto jconfig = Json::Load(StringView{config});
  auto ext_mem_path = OptionalArg<String>(jconfig, "external_memory_path", std::string(""));
  std::cerr << "XGQuantileDMatrixCreateFromCallbackImpl external_memory_path " << ext_mem_path
            << std::endl;

  xgboost::jni::DataIteratorProxy proxy{jdata_iter, fs::path{ext_mem_path}};
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
}  // namespace xgboost::jni
