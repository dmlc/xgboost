/*!
 * Copyright 2017 XGBoost contributors
 */


#include <thrust/fill.h>
#include "./host_device_vector.h"
#include "./device_helpers.cuh"

namespace xgboost {


template <typename T>
struct HostDeviceVectorImpl {
  struct DeviceShard {
    DeviceShard() : index_(-1), device_(-1), start_(0), on_d_(false), vec_(nullptr) {}

    static size_t ShardStart(size_t size, int ndevices, int index) {
      size_t portion = dh::DivRoundUp(size, ndevices);
      size_t begin = index * portion;
      begin = begin > size ? size : begin;
      return begin;
    }

    static size_t ShardSize(size_t size, int ndevices, int index) {
      size_t portion = dh::DivRoundUp(size, ndevices);
      size_t begin = index * portion, end = (index + 1) * portion;
      begin = begin > size ? size : begin;
      end = end > size ? size : end;
      return end - begin;
    }

    void Init(HostDeviceVectorImpl<T>* vec, int device) {
      if (vec_ == nullptr) { vec_ = vec; }
      CHECK_EQ(vec, vec_);
      device_ = device;
      index_ = vec_->devices_.Index(device);
      size_t size_h = vec_->Size();
      int ndevices = vec_->devices_.Size();
      start_ = ShardStart(size_h, ndevices, index_);
      size_t size_d = ShardSize(size_h, ndevices, index_);
      dh::safe_cuda(cudaSetDevice(device_));
      data_.resize(size_d);
      on_d_ = !vec_->on_h_;
    }

    void ScatterFrom(const T* begin) {
      // TODO(canonizer): avoid full copy of host data
      LazySyncDevice();
      dh::safe_cuda(cudaSetDevice(device_));
      dh::safe_cuda(cudaMemcpy(data_.data().get(), begin + start_,
                               data_.size() * sizeof(T), cudaMemcpyDefault));
    }

    void GatherTo(thrust::device_ptr<T> begin) {
      LazySyncDevice();
      dh::safe_cuda(cudaSetDevice(device_));
      dh::safe_cuda(cudaMemcpy(begin.get() + start_, data_.data().get(),
                               data_.size() * sizeof(T), cudaMemcpyDefault));
    }

    void Fill(T v) {
      // TODO(canonizer): avoid full copy of host data
      LazySyncDevice();
      dh::safe_cuda(cudaSetDevice(device_));
      thrust::fill(data_.begin(), data_.end(), v);
    }

    void Copy(DeviceShard* other) {
      // TODO(canonizer): avoid full copy of host data for this (but not for other)
      LazySyncDevice();
      other->LazySyncDevice();
      dh::safe_cuda(cudaSetDevice(device_));
      dh::safe_cuda(cudaMemcpy(data_.data().get(), other->data_.data().get(),
                               data_.size() * sizeof(T), cudaMemcpyDefault));
    }

    void LazySyncHost() {
      dh::safe_cuda(cudaSetDevice(device_));
      thrust::copy(data_.begin(), data_.end(), vec_->data_h_.begin() + start_);
      on_d_ = false;
    }

    void LazySyncDevice() {
      if (on_d_) { return; }
      // data is on the host
      size_t size_h = vec_->data_h_.size();
      int ndevices = vec_->devices_.Size();
      start_ = ShardStart(size_h, ndevices, index_);
      size_t size_d = ShardSize(size_h, ndevices, index_);
      dh::safe_cuda(cudaSetDevice(device_));
      data_.resize(size_d);
      thrust::copy(vec_->data_h_.begin() + start_,
                   vec_->data_h_.begin() + start_ + size_d, data_.begin());
      on_d_ = true;
      // this may cause a race condition if LazySyncDevice() is called
      // from multiple threads in parallel;
      // however, the race condition is benign, and will not cause problems
      vec_->on_h_ = false;
      vec_->size_d_ = vec_->data_h_.size();
    }

    int index_;
    int device_;
    thrust::device_vector<T> data_;
    size_t start_;
    // true if there is an up-to-date copy of data on device, false otherwise
    bool on_d_;
    HostDeviceVectorImpl<T>* vec_;
  };

  HostDeviceVectorImpl(size_t size, T v, GPUSet devices)
    : devices_(devices), on_h_(devices.IsEmpty()), size_d_(0) {
    if (!devices.IsEmpty()) {
      size_d_ = size;
      InitShards();
      Fill(v);
    } else {
      data_h_.resize(size, v);
    }
  }

  // Init can be std::vector<T> or std::initializer_list<T>
  template <class Init>
  HostDeviceVectorImpl(const Init& init, GPUSet devices)
    : devices_(devices), on_h_(devices.IsEmpty()), size_d_(0) {
    if (!devices.IsEmpty()) {
      size_d_ = init.size();
      InitShards();
      Copy(init);
    } else {
      data_h_ = init;
    }
  }

  void InitShards() {
    int ndevices = devices_.Size();
    shards_.resize(ndevices);
    dh::ExecuteIndexShards(&shards_, [&](int i, DeviceShard& shard) {
        shard.Init(this, devices_[i]);
      });
  }

  HostDeviceVectorImpl(const HostDeviceVectorImpl<T>&) = delete;
  HostDeviceVectorImpl(HostDeviceVectorImpl<T>&&) = delete;
  void operator=(const HostDeviceVectorImpl<T>&) = delete;
  void operator=(HostDeviceVectorImpl<T>&&) = delete;

  size_t Size() const { return on_h_ ? data_h_.size() : size_d_; }

  GPUSet Devices() const { return devices_; }

  T* DevicePointer(int device) {
    CHECK(devices_.Contains(device));
    LazySyncDevice(device);
    return shards_[devices_.Index(device)].data_.data().get();
  }

  size_t DeviceSize(int device) {
    CHECK(devices_.Contains(device));
    LazySyncDevice(device);
    return shards_[devices_.Index(device)].data_.size();
  }

  size_t DeviceStart(int device) {
    CHECK(devices_.Contains(device));
    LazySyncDevice(device);
    return shards_[devices_.Index(device)].start_;
  }

  thrust::device_ptr<T> tbegin(int device) {  // NOLINT
    return thrust::device_ptr<T>(DevicePointer(device));
  }

  thrust::device_ptr<T> tend(int device) {  // NOLINT
    return tbegin(device) + DeviceSize(device);
  }

  void ScatterFrom(thrust::device_ptr<T> begin, thrust::device_ptr<T> end) {
    CHECK_EQ(end - begin, Size());
    if (on_h_) {
      thrust::copy(begin, end, data_h_.begin());
    } else {
      dh::ExecuteShards(&shards_, [&](DeviceShard& shard) {
          shard.ScatterFrom(begin.get());
        });
    }
  }

  void GatherTo(thrust::device_ptr<T> begin, thrust::device_ptr<T> end) {
    CHECK_EQ(end - begin, Size());
    if (on_h_) {
      thrust::copy(data_h_.begin(), data_h_.end(), begin);
    } else {
      dh::ExecuteShards(&shards_, [&](DeviceShard& shard) { shard.GatherTo(begin); });
    }
  }

  void Fill(T v) {
    if (on_h_) {
      std::fill(data_h_.begin(), data_h_.end(), v);
    } else {
      dh::ExecuteShards(&shards_, [&](DeviceShard& shard) { shard.Fill(v); });
    }
  }

  void Copy(HostDeviceVectorImpl<T>* other) {
    CHECK_EQ(Size(), other->Size());
    if (on_h_ && other->on_h_) {
      std::copy(other->data_h_.begin(), other->data_h_.end(), data_h_.begin());
    } else {
      CHECK(devices_ == other->devices_);
      dh::ExecuteIndexShards(&shards_, [&](int i, DeviceShard& shard) {
          shard.Copy(&other->shards_[i]);
        });
    }
  }

  void Copy(const std::vector<T>& other) {
    CHECK_EQ(Size(), other.size());
    if (on_h_) {
      std::copy(other.begin(), other.end(), data_h_.begin());
    } else {
      dh::ExecuteShards(&shards_, [&](DeviceShard& shard) {
          shard.ScatterFrom(other.data());
        });
    }
  }

  void Copy(std::initializer_list<T> other) {
    CHECK_EQ(Size(), other.size());
    if (on_h_) {
      std::copy(other.begin(), other.end(), data_h_.begin());
    } else {
      dh::ExecuteShards(&shards_, [&](DeviceShard& shard) {
          shard.ScatterFrom(other.begin());
        });
    }
  }

  std::vector<T>& HostVector() {
    LazySyncHost();
    return data_h_;
  }

  void Reshard(GPUSet new_devices) {
    if (devices_ == new_devices)
      return;
    CHECK(devices_.IsEmpty());
    devices_ = new_devices;
    InitShards();
  }

  void Resize(size_t new_size, T v) {
    if (new_size == Size())
      return;
    if (Size() == 0 && !devices_.IsEmpty()) {
      // fast on-device resize
      on_h_ = false;
      size_d_ = new_size;
      InitShards();
      Fill(v);
    } else {
      // resize on host
      LazySyncHost();
      data_h_.resize(new_size, v);
    }
  }

  void LazySyncHost() {
    if (on_h_)
      return;
    if (data_h_.size() != size_d_)
      data_h_.resize(size_d_);
    dh::ExecuteShards(&shards_, [&](DeviceShard& shard) { shard.LazySyncHost(); });
    on_h_ = true;
  }

  void LazySyncDevice(int device) {
    CHECK(devices_.Contains(device));
    shards_[devices_.Index(device)].LazySyncDevice();
  }

  std::vector<T> data_h_;
  bool on_h_;
  // the total size of the data stored on the devices
  size_t size_d_;
  GPUSet devices_;
  std::vector<DeviceShard> shards_;
};

template <typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, T v, GPUSet devices)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(size, v, devices);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(std::initializer_list<T> init, GPUSet devices)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init, devices);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const std::vector<T>& init, GPUSet devices)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init, devices);
}

template <typename T>
HostDeviceVector<T>::~HostDeviceVector() {
  HostDeviceVectorImpl<T>* tmp = impl_;
  impl_ = nullptr;
  delete tmp;
}

template <typename T>
size_t HostDeviceVector<T>::Size() const { return impl_->Size(); }

template <typename T>
GPUSet HostDeviceVector<T>::Devices() const { return impl_->Devices(); }

template <typename T>
T* HostDeviceVector<T>::DevicePointer(int device) { return impl_->DevicePointer(device); }

template <typename T>
size_t HostDeviceVector<T>::DeviceStart(int device) { return impl_->DeviceStart(device); }

template <typename T>
size_t HostDeviceVector<T>::DeviceSize(int device) { return impl_->DeviceSize(device); }

template <typename T>
thrust::device_ptr<T> HostDeviceVector<T>::tbegin(int device) {  // NOLINT
  return impl_->tbegin(device);
}

template <typename T>
thrust::device_ptr<T> HostDeviceVector<T>::tend(int device) {  // NOLINT
  return impl_->tend(device);
}

template <typename T>
void HostDeviceVector<T>::ScatterFrom
(thrust::device_ptr<T> begin, thrust::device_ptr<T> end) {
  impl_->ScatterFrom(begin, end);
}

template <typename T>
void HostDeviceVector<T>::GatherTo
(thrust::device_ptr<T> begin, thrust::device_ptr<T> end) {
  impl_->GatherTo(begin, end);
}

template <typename T>
void HostDeviceVector<T>::Fill(T v) {
  impl_->Fill(v);
}

template <typename T>
void HostDeviceVector<T>::Copy(HostDeviceVector<T>* other) {
  impl_->Copy(other->impl_);
}

template <typename T>
void HostDeviceVector<T>::Copy(const std::vector<T>& other) {
  impl_->Copy(other);
}

template <typename T>
void HostDeviceVector<T>::Copy(std::initializer_list<T> other) {
  impl_->Copy(other);
}

template <typename T>
std::vector<T>& HostDeviceVector<T>::HostVector() { return impl_->HostVector(); }

template <typename T>
void HostDeviceVector<T>::Reshard(GPUSet new_devices) {
  impl_->Reshard(new_devices);
}

template <typename T>
void HostDeviceVector<T>::Resize(size_t new_size, T v) {
  impl_->Resize(new_size, v);
}

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<GradientPair>;
template class HostDeviceVector<unsigned int>;

}  // namespace xgboost
