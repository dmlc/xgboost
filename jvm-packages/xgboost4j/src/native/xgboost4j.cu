#include <jni.h>
#include <thrust/device_vector.h>

#include "../../../../src/data/array_interface.h"
#include "../../../../src/common/device_helpers.cuh"

extern JavaVM*& GlobalJvm();
extern void setHandle(JNIEnv *jenv, jlongArray jhandle, void* handle);

namespace xgboost {
namespace spark {

template <typename T>
T const* RawPtr(std::vector<T> const& data) {
  return data.data();
}

template <typename T>
T const* RawPtr(thrust::device_vector<T> const& data) {
  return data.data().get();
}

template <typename DCont, typename VCont>
void CopyInterface(std::vector<xgboost::ArrayInterface> const& interface_arr,
                   cudaMemcpyKind kind,
                   std::vector<DCont> *p_data,
                   std::vector<VCont>* p_mask,
                   xgboost::Json* p_out) {
  p_data->resize(interface_arr.size());
  p_mask->resize(interface_arr.size());
  for (size_t c = 0; c < interface_arr.size(); ++c) {
    auto const& interface = interface_arr.at(c);
    size_t element_size = interface.type[2];
    size_t size = element_size * interface.num_rows * interface.num_cols;

    auto& data = (*p_data)[c];
    auto& mask = (*p_mask)[c];
    data.resize(size);
    cudaMemcpyAsync(interface.data, RawPtr(data),
                    size, cudaMemcpyDeviceToHost);

    mask.resize(interface.valid.Size());
    cudaMemcpyAsync(interface.valid.Data(), RawPtr(mask),
                    interface.valid.Size(), kind);

    auto& out = (*p_out)[c];
    out["data"] = Integer(reinterpret_cast<Integer::Int>(RawPtr(data)));
    out["shape"] = Array(
        std::vector<Json>{Json(Integer(interface.num_rows)),
              Json(Integer(interface.num_cols))});

    out["mask"] = Object();
    out["mask"]["data"] = Integer(reinterpret_cast<Integer::Int>(RawPtr(mask)));
    out["mask"]["shape"] = Array(
        std::vector<Json>{Json(Integer(interface.num_rows)),
              Json(Integer(interface.num_cols))});
  }
}

namespace xgboost {
namespace spark {
template <typename DCont, typename VCont>
struct ColumnContainer {
  std::vector<std::vector<DCont>> data;
  std::vector<std::vector<VCont>> valid;
  std::vector<Json> interfaces;

  void Resize(size_t n) {
    data.resize(n);
    valid.resize(n);
    interfaces.resize(n);
  }
};

using ColumnHost = ColumnContainer<std::vector<char>, std::vector<std::uint8_t>>;

class DataIteratorProxy {
  DMatrixHandle proxy_;
  JNIEnv *jenv_;
  int jni_status_;
  jobject jiter_;

  ColumnHost host_columns_;

  size_t it_ {0};
  size_t n_batches_ {0};

 public:
  explicit DataIteratorProxy(jobject jiter) : jiter_{jiter} {
    XGProxyDMatrixCreate(&proxy_);
    jni_status_ =
        GlobalJvm()->GetEnv(reinterpret_cast<void **>(&jenv_), JNI_VERSION_1_6);
    this->InitializeLoop();
    this->Reset();
  }
  ~DataIteratorProxy() {
    XGDMatrixFree(proxy_);
  }

  DMatrixHandle GetDMatrixHandle() const { return proxy_; }

  void InitializeLoop() {
    while (true) {
      try {
        jclass iterClass = jenv_->FindClass("java/util/Iterator");
        jmethodID has_next = jenv_->GetMethodID(iterClass, "hasNext", "()Z");
        jmethodID next =
            jenv_->GetMethodID(iterClass, "next", "()Ljava/lang/Object;");
        if (jenv_->CallBooleanMethod(jiter_, has_next)) {
          jobject batch = jenv_->CallObjectMethod(jiter_, next);
          if (!batch) {
            CHECK(jenv_->ExceptionOccurred());
            jenv_->ExceptionDescribe();
          }
          jclass batch_class = jenv_->GetObjectClass(batch);
          CHECK(batch_class);
          jmethodID get_array_interface = jenv_->GetMethodID(
              batch_class, "getArrayInterface", "()Ljava/lang/Object;");
          CHECK(get_array_interface);

          auto jinterface = static_cast<jstring>(
              jenv_->CallObjectMethod(batch, get_array_interface));
          CHECK(jinterface);
          char const *c_interface_str =
              jenv_->GetStringUTFChars(jinterface, nullptr);
          CHECK(c_interface_str);
          std::string interface_str {c_interface_str};
          jenv_->ReleaseStringUTFChars(jinterface, c_interface_str);

          ++n_batches_;
          host_columns_.Resize(n_batches_);

          auto json_interface = Json::Load({interface_str.c_str(), interface_str.size()});
          auto json_columns = get<Array const>(json_interface);
          std::vector<ArrayInterface> interfaces(get<Array const>(json_interface).size());

          for (auto& json_col : json_columns) {
            auto column = ArrayInterface(get<Object const>(json_col));
            interfaces.emplace_back(column);
          }

          host_columns_.interfaces.back() = json_interface;
          CopyInterface(interfaces,
                        cudaMemcpyDeviceToHost,
                        &host_columns_.data.back(),
                        &host_columns_.valid.back(),
                        &host_columns_.interfaces.back());
        } else {
          break;
        }
      } catch (dmlc::Error const &e) {
        if (jni_status_ == JNI_EDETACHED) {
          GlobalJvm()->DetachCurrentThread();
        }
        LOG(FATAL) << e.what();
      }
    }
  }

  void Reset()  {
    it_ = 0;
  }

  int Next() {
    if (it_ == n_batches_) {
      return 0;
    }
    auto json_interface = host_columns_.interfaces.at(it_);
    auto json_columns = get<Array const>(json_interface);

    std::vector<ArrayInterface> in(get<Array const>(json_interface).size());
    for (auto& json_col : json_columns) {
      auto column = ArrayInterface(get<Object const>(json_col));
      in.emplace_back(column);
    }

    std::string temp;
    Json::Dump(json_interface, &temp);
    Json out { Json::Load({temp.c_str(), temp.size()}) };

    std::vector<thrust::device_vector<char>> data;
    std::vector<thrust::device_vector<uint8_t>> mask;
    CopyInterface(in, cudaMemcpyHostToDevice, &data, &mask, &out);

    std::string interface_str;
    Json::Dump(out, &interface_str);
    XGDMatrixSetDataCudaArrayInterface(proxy_, interface_str.c_str());
    it_++;
    return 1;
  };
};
}  // namespace spark
}  // namespace xgboost

namespace {
void Reset(DataIterHandle self) {
  static_cast<xgboost::spark::DataIteratorProxy*>(self)->Reset();
}

int Next(DataIterHandle self) {
  return static_cast<xgboost::spark::DataIteratorProxy*>(self)->Next();
}
}  // anonymous namespace

jint XGDeviceQuantileDMatrixCreateFromCallbackImpl(JNIEnv *jenv, jclass jcls,
                                                   jobject jiter,
                                                   jfloat jmissing,
                                                   jint jmax_bin, jint jnthread,
                                                   jlongArray jout) {
  xgboost::spark::DataIteratorProxy proxy(jiter);
  DMatrixHandle result;
  auto ret =
      XGDMatrixCreateFromCallback(&proxy, proxy.GetDMatrixHandle(), Reset, Next,
                                  jmissing, jnthread, jmax_bin, &result);
  setHandle(jenv, jout, result);
  return ret;
}

}  // namespace spark
}  // namespace xgboost
