/*!
 * Copyright 2017-2023 by Contributors
 * \file hist_util.h
 */
#ifndef PLUGIN_SYCL_COMMON_HIST_UTIL_H_
#define PLUGIN_SYCL_COMMON_HIST_UTIL_H_

#include <vector>
#include <unordered_map>
#include <memory>

#include "../data.h"
#include "row_set.h"

#include "../../src/common/hist_util.h"
#include "../data/gradient_index.h"

#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

template<typename GradientSumT, MemoryType memory_type = MemoryType::shared>
using GHistRow = USMVector<xgboost::detail::GradientPairInternal<GradientSumT>, memory_type>;

using BinTypeSize = ::xgboost::common::BinTypeSize;

class ColumnMatrix;

/*!
 * \brief Fill histogram with zeroes
 */
template<typename GradientSumT>
void InitHist(::sycl::queue qu,
              GHistRow<GradientSumT, MemoryType::on_device>* hist,
              size_t size, ::sycl::event* event);

/*!
 * \brief Copy histogram from src to dst
 */
template<typename GradientSumT>
void CopyHist(::sycl::queue qu,
              GHistRow<GradientSumT, MemoryType::on_device>* dst,
              const GHistRow<GradientSumT, MemoryType::on_device>& src,
              size_t size);

/*!
 * \brief Compute subtraction: dst = src1 - src2
 */
template<typename GradientSumT>
::sycl::event SubtractionHist(::sycl::queue qu,
                              GHistRow<GradientSumT, MemoryType::on_device>* dst,
                              const GHistRow<GradientSumT, MemoryType::on_device>& src1,
                              const GHistRow<GradientSumT, MemoryType::on_device>& src2,
                              size_t size, ::sycl::event event_priv);

/*!
 * \brief Histograms of gradient statistics for multiple nodes
 */
template<typename GradientSumT, MemoryType memory_type = MemoryType::shared>
class HistCollection {
 public:
  using GHistRowT = GHistRow<GradientSumT, memory_type>;

  // Access histogram for i-th node
  GHistRowT& operator[](bst_uint nid) {
    return *(data_.at(nid));
  }

  const GHistRowT& operator[](bst_uint nid) const {
    return *(data_.at(nid));
  }

  // Initialize histogram collection
  void Init(::sycl::queue qu, uint32_t nbins) {
    qu_ = qu;
    if (nbins_ != nbins) {
      nbins_ = nbins;
      data_.clear();
    }
  }

  // Create an empty histogram for i-th node
  ::sycl::event AddHistRow(bst_uint nid) {
    ::sycl::event event;
    if (data_.count(nid) == 0) {
      data_[nid] =
        std::make_shared<GHistRowT>(&qu_, nbins_,
                                    xgboost::detail::GradientPairInternal<GradientSumT>(0, 0),
                                    &event);
    } else {
      data_[nid]->Resize(&qu_, nbins_,
                         xgboost::detail::GradientPairInternal<GradientSumT>(0, 0),
                         &event);
    }
    return event;
  }

 private:
  /*! \brief Number of all bins over all features */
  uint32_t nbins_ = 0;

  std::unordered_map<uint32_t, std::shared_ptr<GHistRowT>> data_;

  ::sycl::queue qu_;
};

/*!
 * \brief Stores temporary histograms to compute them in parallel
 */
template<typename GradientSumT>
class ParallelGHistBuilder {
 public:
  using GHistRowT = GHistRow<GradientSumT, MemoryType::on_device>;

  void Init(::sycl::queue qu, size_t nbins) {
    qu_ = qu;
    if (nbins != nbins_) {
      hist_buffer_.Init(qu_, nbins);
      nbins_ = nbins;
    }
  }

  void Reset(size_t nblocks) {
    hist_device_buffer_.Resize(&qu_, nblocks * nbins_ * 2);
  }

  GHistRowT& GetDeviceBuffer() {
    return hist_device_buffer_;
  }

 protected:
  /*! \brief Number of bins in each histogram */
  size_t nbins_ = 0;
  /*! \brief Buffers for histograms for all nodes processed */
  HistCollection<GradientSumT> hist_buffer_;

  /*! \brief Buffer for additional histograms for Parallel processing  */
  GHistRowT hist_device_buffer_;

  ::sycl::queue qu_;
};

/*!
 * \brief Builder for histograms of gradient statistics
 */
template<typename GradientSumT>
class GHistBuilder {
 public:
  template<MemoryType memory_type = MemoryType::shared>
  using GHistRowT = GHistRow<GradientSumT, memory_type>;

  GHistBuilder() = default;
  GHistBuilder(::sycl::queue qu, uint32_t nbins) : qu_{qu}, nbins_{nbins} {}

  // Construct a histogram via histogram aggregation
  ::sycl::event BuildHist(const USMVector<GradientPair, MemoryType::on_device>& gpair_device,
                          const RowSetCollection::Elem& row_indices,
                          const GHistIndexMatrix& gmat,
                          GHistRowT<MemoryType::on_device>* HistCollection,
                          bool isDense,
                          GHistRowT<MemoryType::on_device>* hist_buffer,
                          ::sycl::event event,
                          bool force_atomic_use = false);

  // Construct a histogram via subtraction trick
  void SubtractionTrick(GHistRowT<MemoryType::on_device>* self,
                        const GHistRowT<MemoryType::on_device>& sibling,
                        const GHistRowT<MemoryType::on_device>& parent);

  uint32_t GetNumBins() const {
      return nbins_;
  }

 private:
  /*! \brief Number of all bins over all features */
  uint32_t nbins_ { 0 };

  ::sycl::queue qu_;
};
}  // namespace common
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_COMMON_HIST_UTIL_H_
