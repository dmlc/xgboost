/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#include <jni.h>
#include <xgboost/c_api.h>

#include "../../../../src/data/array_interface.h"
#include "jvm_utils.h"

namespace xgboost {
namespace jni {

template <typename T> T CheckJvmCall(T const &v, JNIEnv *jenv) {
  if (!v) {
    CHECK(jenv->ExceptionOccurred());
    jenv->ExceptionDescribe();
  }
  return v;
}

class ExternalMemoryIteratorProxy {
  jobject jiter_;
  JNIEnv *jenv_;
  DMatrixHandle proxy_;
  int jni_status_;
  jobject last_batch_ {nullptr};

 public:
  explicit ExternalMemoryIteratorProxy(jobject jiter): jiter_(jiter) {
    XGProxyDMatrixCreate(&proxy_);
    jni_status_ =
        GlobalJvm()->GetEnv(reinterpret_cast<void **>(&jenv_), JNI_VERSION_1_6);
  }

  ~ExternalMemoryIteratorProxy() {
    XGDMatrixFree(proxy_);
  }

  DMatrixHandle GetDMatrixHandle() const { return proxy_; }

  void CloseJvmBatch() {
    if (last_batch_) {
      jclass batch_class = CheckJvmCall(jenv_->GetObjectClass(last_batch_), jenv_);
      jmethodID closeMethod = CheckJvmCall(jenv_->GetMethodID(batch_class, "close", "()V"), jenv_);
      jenv_->CallVoidMethod(last_batch_, closeMethod);
      last_batch_ = nullptr;
    }
  }

  void SetArrayInterface(std::string interface_str) {
    auto json_interface =
        Json::Load({interface_str.c_str(), interface_str.size()});
    CHECK(!IsA<Null>(json_interface));

    std::string str;
    Json features = json_interface["features"];
    Json::Dump(features, &str);
    XGProxyDMatrixSetDataCudaColumnar(proxy_, str.c_str());

    // set the meta info.
    auto json_map = get<Object const>(json_interface);
    if (json_map.find("label") == json_map.cend()) {
      LOG(FATAL) << "Must have a label field.";
    }
    Json label = json_interface["label"];
    CHECK(!IsA<Null>(label));
    Json::Dump(label, &str);
    XGDMatrixSetInfoFromInterface(proxy_, "label", str.c_str());

    if (json_map.find("weight") != json_map.cend()) {
      Json weight = json_interface["weight"];
      CHECK(!IsA<Null>(weight));
      Json::Dump(weight, &str);
      XGDMatrixSetInfoFromInterface(proxy_, "weight", str.c_str());
    }

    if (json_map.find("baseMargin") != json_map.cend()) {
      Json basemargin = json_interface["baseMargin"];
      Json::Dump(basemargin, &str);
      XGDMatrixSetInfoFromInterface(proxy_, "base_margin", str.c_str());
    }

    if (json_map.find("qid") != json_map.cend()) {
      Json qid = json_interface["qid"];
      Json::Dump(qid, &str);
      XGDMatrixSetInfoFromInterface(proxy_, "qid", str.c_str());
    }
  }

  int Next() {
    try {
      this->CloseJvmBatch();
      jclass iterClass = jenv_->FindClass("java/util/Iterator");
      jmethodID has_next = CheckJvmCall(jenv_->GetMethodID(iterClass, "hasNext", "()Z"), jenv_);
      jmethodID next = CheckJvmCall(
          jenv_->GetMethodID(iterClass, "next", "()Ljava/lang/Object;"), jenv_);

      if (jenv_->CallBooleanMethod(jiter_, has_next)) {
        // batch should be ColumnBatch from jvm
        jobject batch = CheckJvmCall(jenv_->CallObjectMethod(jiter_, next), jenv_);
        jclass batch_class = CheckJvmCall(jenv_->GetObjectClass(batch), jenv_);
        jmethodID toJson = CheckJvmCall(
            jenv_->GetMethodID(batch_class, "toJson", "()Ljava/lang/String;"), jenv_);

        auto jinterface = static_cast<jstring>(jenv_->CallObjectMethod(batch, toJson));
        CheckJvmCall(jinterface, jenv_);
        char const *c_interface_str = CheckJvmCall(
            jenv_->GetStringUTFChars(jinterface, nullptr), jenv_);
        this->SetArrayInterface(c_interface_str);
        jenv_->ReleaseStringUTFChars(jinterface, c_interface_str);
        last_batch_ = batch;
        return 1;
      } else {
        return 0;
      }
    } catch (dmlc::Error const &e) {
      if (jni_status_ == JNI_EDETACHED) {
        GlobalJvm()->DetachCurrentThread();
      }
      LOG(FATAL) << e.what();
    }
    return 0;
  }

  void Reset() {
    this->CloseJvmBatch();
  }
};

namespace {
void Reset(DataIterHandle self) {
  static_cast<xgboost::jni::ExternalMemoryIteratorProxy *>(self)->Reset();
}

int Next(DataIterHandle self) {
  return static_cast<xgboost::jni::ExternalMemoryIteratorProxy *>(self)->Next();
}

template <typename T>
using Deleter = std::function<void(T *)>;
} // anonymous namespace

XGB_DLL int XGQuantileDMatrixCreateFromCallbackImpl(JNIEnv *jenv, jclass, jobject jdata_iter,
                                                    jlongArray jref, char const *config,
                                                    jlongArray jout) {
  xgboost::jni::ExternalMemoryIteratorProxy proxy(jdata_iter);
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
