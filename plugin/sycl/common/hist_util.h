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
 * \brief Compute subtraction: dst = src1 - src2
 */
template<typename GradientSumT>
::sycl::event SubtractionHist(::sycl::queue qu,
                              GHistRow<GradientSumT, MemoryType::on_device>* dst,
                              const GHistRow<GradientSumT, MemoryType::on_device>& src1,
                              const GHistRow<GradientSumT, MemoryType::on_device>& src2,
                              size_t size, ::sycl::event event_priv);

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
