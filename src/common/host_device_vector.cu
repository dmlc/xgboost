/*!
 * Copyright 2017 XGBoost contributors
 */

#include "./host_device_vector.h"
#include <thrust/fill.h>
#include <xgboost/data.h>
#include "./device_helpers.cuh"


namespace xgboost {


template <typename T>
struct HostDeviceVectorImpl {
  struct DeviceShard {
    DeviceShard()
      : index_(-1), proper_size_(0), device_(-1), start_(0), on_d_(false), cached_size_(~0),
        vec_(nullptr) {}

    void Init(HostDeviceVectorImpl<T>* vec, int device) {
      if (vec_ == nullptr) { vec_ = vec; }
      CHECK_EQ(vec, vec_);
      device_ = device;
      index_ = vec_->distribution_.devices_.Index(device);
      LazyResize(vec_->Size());
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
                               proper_size_ * sizeof(T), cudaMemcpyDefault));
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
      dh::safe_cuda(cudaMemcpy(vec_->data_h_.data() + start_,
                               data_.data().get(),  proper_size_ * sizeof(T),
                               cudaMemcpyDeviceToHost));
      on_d_ = false;
    }

    void LazyResize(size_t new_size) {
      if (new_size == cached_size_) { return; }
      // resize is required
      int ndevices = vec_->distribution_.devices_.Size();
      start_ = vec_->distribution_.ShardStart(new_size, index_);
      proper_size_ = vec_->distribution_.ShardProperSize(new_size, index_);
      size_t size_d = vec_->distribution_.ShardSize(new_size, index_);
      dh::safe_cuda(cudaSetDevice(device_));
      data_.resize(size_d);
      cached_size_ = new_size;
    }

    void LazySyncDevice() {
      if (on_d_) { return; }
      // data is on the host
      size_t size_h = vec_->data_h_.size();
      LazyResize(size_h);
      dh::safe_cuda(cudaSetDevice(device_));
      dh::safe_cuda(
          cudaMemcpy(data_.data().get(), vec_->data_h_.data() + start_,
                     data_.size() * sizeof(T), cudaMemcpyHostToDevice));
      on_d_ = true;
      // this may cause a race condition if LazySyncDevice() is called
      // from multiple threads in parallel;
      // however, the race condition is benign, and will not cause problems
      vec_->on_h_ = false;
      vec_->size_d_ = size_h;
    }

    int index_;
    int device_;
    thrust::device_vector<T> data_;
    // cached vector size
    size_t cached_size_;
    size_t start_;
    // size of the portion to copy back to the host
    size_t proper_size_;
    // true if there is an up-to-date copy of data on device, false otherwise
    bool on_d_;
    HostDeviceVectorImpl<T>* vec_;
  };

  HostDeviceVectorImpl(size_t size, T v, GPUDistribution distribution)
    : distribution_(distribution), on_h_(distribution.IsEmpty()), size_d_(0) {
    if (!distribution_.IsEmpty()) {
      size_d_ = size;
      InitShards();
      Fill(v);
    } else {
      data_h_.resize(size, v);
    }
  }

  // Init can be std::vector<T> or std::initializer_list<T>
  template <class Init>
  HostDeviceVectorImpl(const Init& init, GPUDistribution distribution)
    : distribution_(distribution), on_h_(distribution.IsEmpty()), size_d_(0) {
    if (!distribution_.IsEmpty()) {
      size_d_ = init.size();
      InitShards();
      Copy(init);
    } else {
      data_h_ = init;
    }
  }

  void InitShards() {
    int ndevices = distribution_.devices_.Size();
    shards_.resize(ndevices);
    dh::ExecuteIndexShards(&shards_, [&](int i, DeviceShard& shard) {
        shard.Init(this, distribution_.devices_[i]);
      });
  }

  size_t Size() const { return on_h_ ? data_h_.size() : size_d_; }

  GPUSet Devices() const { return distribution_.devices_; }

  const GPUDistribution& Distribution() const { return distribution_; }

  T* DevicePointer(int device) {
    CHECK(distribution_.devices_.Contains(device));
    LazySyncDevice(device);
    return shards_[distribution_.devices_.Index(device)].data_.data().get();
  }

  common::Span<T> DeviceSpan(int device) {
    CHECK(devices_.Contains(device));
    LazySyncDevice(device);
    return { shards_[devices_.Index(device)].data_.data().get(),
             static_cast<typename common::Span<T>::index_type>(Size()) };
  }

  size_t DeviceSize(int device) {
    CHECK(distribution_.devices_.Contains(device));
    LazySyncDevice(device);
    return shards_[distribution_.devices_.Index(device)].data_.size();
  }

  size_t DeviceStart(int device) {
    CHECK(distribution_.devices_.Contains(device));
    LazySyncDevice(device);
    return shards_[distribution_.devices_.Index(device)].start_;
  }

  thrust::device_ptr<T> tbegin(int device) {  // NOLINT
    return thrust::device_ptr<T>(DevicePointer(device));
  }

  thrust::device_ptr<T> tend(int device) {  // NOLINT
    return tbegin(device) + DeviceSize(device);
  }

  void ScatterFrom(thrust::device_ptr<const T> begin, thrust::device_ptr<const T> end) {
    CHECK_EQ(end - begin, Size());
    if (on_h_) {
      dh::safe_cuda(cudaMemcpy(data_h_.data(), begin.get(),
                               (end - begin) * sizeof(T),
                               cudaMemcpyDeviceToHost));
    } else {
      dh::ExecuteShards(&shards_, [&](DeviceShard& shard) {
        shard.ScatterFrom(begin.get());
      });
    }
  }

  void GatherTo(thrust::device_ptr<T> begin, thrust::device_ptr<T> end) {
    CHECK_EQ(end - begin, Size());
    if (on_h_) {
      dh::safe_cuda(cudaMemcpy(begin.get(), data_h_.data(),
                               data_h_.size() * sizeof(T),
                               cudaMemcpyHostToDevice));
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
      CHECK(distribution_ == other->distribution_);
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

  void Reshard(const GPUDistribution& distribution) {
    if (distribution_ == distribution)
      return;
    CHECK(distribution_.IsEmpty());
    distribution_ = distribution;
    InitShards();
  }

  void Reshard(GPUSet new_devices) {
    if (distribution_.Devices() == new_devices)
      return;
    Reshard(GPUDistribution::Block(new_devices));
  }

  void Resize(size_t new_size, T v) {
    if (new_size == Size())
      return;
    if (distribution_.IsFixedSize()) {
      CHECK_EQ(new_size, distribution_.offsets_.back());
    }
    if (Size() == 0 && !distribution_.IsEmpty()) {
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
    GPUSet devices = distribution_.Devices();
    CHECK(devices.Contains(device));
    shards_[devices.Index(device)].LazySyncDevice();
  }

  std::vector<T> data_h_;
  bool on_h_;
  // the total size of the data stored on the devices
  size_t size_d_;
  GPUDistribution distribution_;
  std::vector<DeviceShard> shards_;
};

template <typename T>
HostDeviceVector<T>::HostDeviceVector
(size_t size, T v, GPUDistribution distribution) : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(size, v, distribution);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector
(std::initializer_list<T> init, GPUDistribution distribution) : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init, distribution);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector
(const std::vector<T>& init, GPUDistribution distribution) : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init, distribution);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const HostDeviceVector<T>& other)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(*other.impl_);
}

template <typename T>
HostDeviceVector<T>& HostDeviceVector<T>::operator=
(const HostDeviceVector<T>& other) {
  if (this == &other) {
    return *this;
  }
  delete impl_;
  impl_ = new HostDeviceVectorImpl<T>(*other.impl_);
  return *this;
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
const GPUDistribution& HostDeviceVector<T>::Distribution() const {
  return impl_->Distribution();
}

template <typename T>
T* HostDeviceVector<T>::DevicePointer(int device) { return impl_->DevicePointer(device); }

template <typename T>
common::Span<T> HostDeviceVector<T>::DeviceSpan(int device) {
  return impl_->DeviceSpan(device);
}

const T* HostDeviceVector<T>::ConstDevicePointer(int device) const {
  return impl_->DevicePointer(device);
}

template <typename T>
size_t HostDeviceVector<T>::DeviceStart(int device) { return impl_->DeviceStart(device); }

const T* HostDeviceVector<T>::DevicePointer(int device) const {
  return impl_->DevicePointer(device);
}

template <typename T>
size_t HostDeviceVector<T>::DeviceSize(int device) const {
  return impl_->DeviceSize(device);
}

template <typename T>
thrust::device_ptr<T> HostDeviceVector<T>::tbegin(int device) {  // NOLINT
  return impl_->tbegin(device);
}

template <typename T>
thrust::device_ptr<const T> HostDeviceVector<T>::tcbegin(int device) const {  // NOLINT
  return thrust::device_ptr<const T>(impl_->tbegin(device));
}

template <typename T>
thrust::device_ptr<T> HostDeviceVector<T>::tend(int device) {  // NOLINT
  return impl_->tend(device);
}

template <typename T>
thrust::device_ptr<const T> HostDeviceVector<T>::tcend(int device) const {  // NOLINT
  return thrust::device_ptr<const T>(impl_->tend(device));
}

template <typename T>
void HostDeviceVector<T>::ScatterFrom
(thrust::device_ptr<const T> begin, thrust::device_ptr<const T> end) {
  impl_->ScatterFrom(begin, end);
}

template <typename T>
void HostDeviceVector<T>::GatherTo
(thrust::device_ptr<T> begin, thrust::device_ptr<T> end) const {
  impl_->GatherTo(begin, end);
}

template <typename T>
void HostDeviceVector<T>::Fill(T v) {
  impl_->Fill(v);
}

template <typename T>
void HostDeviceVector<T>::Copy(const HostDeviceVector<T>& other) {
  impl_->Copy(other.impl_);
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
const std::vector<T>& HostDeviceVector<T>::ConstHostVector() const {
  return impl_->HostVector();
}

template <typename T>
void HostDeviceVector<T>::Reshard(GPUSet new_devices) const {
  impl_->Reshard(new_devices);
}

template <typename T>
void HostDeviceVector<T>::Reshard(const GPUDistribution& distribution) const {
  impl_->Reshard(distribution);
}

template <typename T>
void HostDeviceVector<T>::Resize(size_t new_size, T v) {
  impl_->Resize(new_size, v);
}

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<GradientPair>;
template class HostDeviceVector<unsigned int>;
template class HostDeviceVector<int>;
template class HostDeviceVector<Entry>;
template class HostDeviceVector<size_t>;

}  // namespace xgboost
