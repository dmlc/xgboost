/*!
 * Copyright by Contributors 2017-2020
 */
#ifndef XGBOOST_COMMON_DATA_ONEAPI_H_
#define XGBOOST_COMMON_DATA_ONEAPI_H_

#include <cstddef>
#include <limits>
#include <mutex>

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/logging.h"
#include "xgboost/host_device_vector.h"

#include "CL/sycl.hpp"

namespace xgboost {

template <typename T>
class USMDeleter {
public:
  explicit USMDeleter(cl::sycl::queue qu) : qu_(qu) {}

  void operator()(T* data) const {
    cl::sycl::free(data, qu_);    
  }

private:
  cl::sycl::queue qu_;
};


template <typename T>
class USMVector {
  static_assert(std::is_standard_layout<T>::value, "USMVector admits only POD types");

public:
  USMVector() : size_(0), data_(nullptr) {}

  USMVector(cl::sycl::queue qu) : qu_(qu), size_(0), data_(nullptr) {}

  USMVector(cl::sycl::queue qu, size_t size) : qu_(qu), size_(size) {
    data_ = std::shared_ptr<T>(cl::sycl::malloc_shared<T>(size_, qu_), USMDeleter<T>(qu_));
  }

  USMVector(cl::sycl::queue qu, size_t size, T v) : qu_(qu), size_(size) {
    data_ = std::shared_ptr<T>(cl::sycl::malloc_shared<T>(size_, qu_), USMDeleter<T>(qu_));
    std::fill(data_.get(), data_.get() + size_, v);
    // switch to handler fill after moving to new compiler
    /*
    qu.submit([&](cl::sycl::handler& cgh) {
      cgh.fill(data_.get(), v, size_);
    }).wait();
    */
  }

  USMVector(cl::sycl::queue qu, const std::vector<T> &vec) : qu_(qu) {
    size_ = vec.size();
    data_ = std::shared_ptr<T>(cl::sycl::malloc_shared<T>(size_, qu_), USMDeleter<T>(qu_));
    std::copy(vec.begin (), vec.end (), data_.get());
  }

  USMVector(const USMVector<T>& other) : qu_(other.qu_), size_(other.size_), data_(other.data_) {
  }

  ~USMVector() {
  }

  USMVector<T>& operator=(const USMVector<T>& other) {
    qu_ = other.qu_;
    size_ = other.size_;
    data_ = other.data_;
    return *this;
  }

  T* Data() { return data_.get(); }
  const T* DataConst() const { return data_.get(); }

  size_t Size() const { return size_; }

  T& operator[] (size_t i) { return data_.get()[i]; }
  const T& operator[] (size_t i) const { return data_.get()[i]; }

  T* Begin () const { return data_.get(); }
  T* End () const { return data_.get() + size_; }

  bool Empty() const { return (size_ == 0); }

  void Clear() {
    data_.reset();
    size_ = 0;
  }

  void Resize(cl::sycl::queue qu, size_t new_size) {
    qu_ = qu;
    if (new_size <= size_) {
      size_ = new_size;
    } else {
      size_t old_size = size_;
      auto data_old = data_;
      size_ = new_size;
      data_ = std::shared_ptr<T>(cl::sycl::malloc_shared<T>(size_, qu_), USMDeleter<T>(qu_));
      if (old_size > 0) {
        std::copy(data_old.get(), data_old.get() + old_size, data_.get());
        // switch to handler copy after moving to new compiler
        /*
        qu.submit([&](cl::sycl::handler& cgh) {
          cgh.copy(data_old.get(), data_.get(), old_size);
        }).wait();
        */
      }
    }
  }

  void Resize(cl::sycl::queue qu, size_t new_size, T v) {
    qu_ = qu;
    if (new_size <= size_) {
      size_ = new_size;
    } else {
      size_t old_size = size_;
      auto data_old = data_;
      size_ = new_size;
      data_ = std::shared_ptr<T>(cl::sycl::malloc_shared<T>(size_, qu_), USMDeleter<T>(qu_));
      if (old_size > 0) {
        std::copy(data_old.get(), data_old.get() + old_size, data_.get());
        // switch to handler copy after moving to new compiler
        /*
        qu.submit([&](cl::sycl::handler& cgh) {
          cgh.copy(data_old.get(), data_.get(), old_size);
        }).wait();
        */
      }
      if (new_size > old_size) {
        std::fill(data_.get() + old_size, data_.get() + new_size, v);
      }
    }
  }

  void Init(cl::sycl::queue qu, const std::vector<T> &vec) {
    qu_ = qu;
    size_ = vec.size();
    data_ = std::shared_ptr<T>(cl::sycl::malloc_shared<T>(size_, qu_), USMDeleter<T>(qu_));
    std::copy(vec.begin (), vec.end (), data_.get());
  }

  using value_type = T;  // NOLINT

private:
  cl::sycl::queue qu_;
  size_t size_;
  std::shared_ptr<T> data_;
};

/*! \brief Element from a sparse vector */
struct EntryOneAPI {
  /*! \brief feature index */
  bst_feature_t index;
  /*! \brief feature value */
  bst_float fvalue;
  /*! \brief default constructor */
  EntryOneAPI() = default;
  /*!
   * \brief constructor with index and value
   * \param index The feature or row index.
   * \param fvalue The feature value.
   */
  EntryOneAPI(bst_feature_t index, bst_float fvalue) : index(index), fvalue(fvalue) {}

  EntryOneAPI(const Entry& entry) : index(entry.index), fvalue(entry.fvalue) {}

  /*! \brief reversely compare feature values */
  inline static bool CmpValue(const EntryOneAPI& a, const EntryOneAPI& b) {
    return a.fvalue < b.fvalue;
  }
  inline bool operator==(const EntryOneAPI& other) const {
    return (this->index == other.index && this->fvalue == other.fvalue);
  }
};

struct DeviceMatrixOneAPI {
  DMatrix* p_mat;  // Pointer to the original matrix on the host
  cl::sycl::queue qu_;
  USMVector<size_t> row_ptr;
  USMVector<EntryOneAPI> data;
  size_t total_offset;

  DeviceMatrixOneAPI(cl::sycl::queue qu, DMatrix* dmat) : p_mat(dmat), qu_(qu) {
    size_t num_row = 0;
    size_t num_nonzero = 0;
    for (auto &batch : dmat->GetBatches<SparsePage>()) {
      const auto& data_vec = batch.data.HostVector();
      const auto& offset_vec = batch.offset.HostVector();
      num_nonzero += data_vec.size();
      num_row += batch.Size();
    }

    row_ptr.Resize(qu_, num_row + 1);
    data.Resize(qu_, num_nonzero);

    size_t data_offset = 0;
    for (auto &batch : dmat->GetBatches<SparsePage>()) {
      const auto& data_vec = batch.data.HostVector();
      const auto& offset_vec = batch.offset.HostVector();
      size_t batch_size = batch.Size();
      if (batch_size > 0) {
        std::copy(offset_vec.data(), offset_vec.data() + batch_size,
                  row_ptr.Data() + batch.base_rowid);
        if (batch.base_rowid > 0) {
          for(size_t i = 0; i < batch_size; i++)
            row_ptr[i + batch.base_rowid] += batch.base_rowid;
        }
        std::copy(data_vec.data(), data_vec.data() + offset_vec[batch_size],
                  data.Data() + data_offset);
        data_offset += offset_vec[batch_size];
      }
    }
    row_ptr[num_row] = data_offset;
    total_offset = data_offset;
  }

  ~DeviceMatrixOneAPI() {
  }
};

}  // namespace xgboost

#endif