/*!
 * Copyright 2017-2020 by Contributors
 * \file hist_util.cc
 */
#include <dmlc/timer.h>
#include <dmlc/omp.h>

#include <rabit/rabit.h>
#include <numeric>
#include <vector>

#include "xgboost/base.h"
#include "../common/common.h"
#include "hist_util.h"
#include "hist_builder.h"
#include "random.h"
#include "column_matrix.h"
#include "quantile.h"
#include "../data/gradient_index.h"

namespace xgboost {
namespace common {

constexpr size_t Prefetch::kNoPrefetchSize;

HistogramCuts::HistogramCuts() {
  cut_ptrs_.HostVector().emplace_back(0);
}

/*!
 * \brief fill a histogram by zeros in range [begin, end)
 */
template<typename GradientSumT>
void InitilizeHistByZeroes(GHistRow<GradientSumT> hist, size_t begin, size_t end) {
#if defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  std::fill(hist.begin() + begin, hist.begin() + end,
            xgboost::detail::GradientPairInternal<GradientSumT>());
#else  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  memset(hist.data() + begin, '\0', (end-begin)*
         sizeof(xgboost::detail::GradientPairInternal<GradientSumT>));
#endif  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
}
template void InitilizeHistByZeroes(GHistRow<float> hist, size_t begin,
                                    size_t end);
template void InitilizeHistByZeroes(GHistRow<double> hist, size_t begin,
                                    size_t end);

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
template<typename GradientSumT>
void IncrementHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> add,
                   size_t begin, size_t end) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.data());
  const GradientSumT* padd = reinterpret_cast<const GradientSumT*>(add.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] += padd[i];
  }
}
template void IncrementHist(GHistRow<float> dst, const GHistRow<float> add,
                            size_t begin, size_t end);
template void IncrementHist(GHistRow<double> dst, const GHistRow<double> add,
                            size_t begin, size_t end);

/*!
 * \brief Copy hist from src to dst in range [begin, end)
 */
template<typename GradientSumT>
void CopyHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> src,
              size_t begin, size_t end) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.data());
  const GradientSumT* psrc = reinterpret_cast<const GradientSumT*>(src.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc[i];
  }
}
template void CopyHist(GHistRow<float> dst, const GHistRow<float> src,
                       size_t begin, size_t end);
template void CopyHist(GHistRow<double> dst, const GHistRow<double> src,
                       size_t begin, size_t end);

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
template<typename GradientSumT>
void SubtractionHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> src1,
                     const GHistRow<GradientSumT> src2,
                     size_t begin, size_t end) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.data());
  const GradientSumT* psrc1 = reinterpret_cast<const GradientSumT*>(src1.data());
  const GradientSumT* psrc2 = reinterpret_cast<const GradientSumT*>(src2.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc1[i] - psrc2[i];
  }
}
template void SubtractionHist(GHistRow<float> dst, const GHistRow<float> src1,
                              const GHistRow<float> src2,
                              size_t begin, size_t end);
template void SubtractionHist(GHistRow<double> dst, const GHistRow<double> src1,
                              const GHistRow<double> src2,
                              size_t begin, size_t end);

template<typename GradientSumT>
void ClearHist(GradientSumT* dest_hist,
                size_t begin, size_t end) {
  for (size_t bin_id = begin; bin_id < end; ++bin_id) {
    dest_hist[bin_id]  = 0;
  }
}
template void ClearHist(float* dest_hist,
                        size_t begin, size_t end);
template void ClearHist(double* dest_hist,
                        size_t begin, size_t end);

template<typename GradientSumT>
void ReduceHist(GradientSumT* dest_hist,
                const std::vector<std::vector<uint16_t>>& local_threads_mapping,
                std::vector<std::vector<std::vector<GradientSumT>>>* histograms,
                const size_t node_id,
                const std::vector<uint16_t>& threads_id_for_node,
                size_t begin, size_t end) {
  const size_t first_thread_id = threads_id_for_node[0];
  CHECK_LT(node_id, local_threads_mapping[first_thread_id].size());
  const size_t mapped_nod_id = local_threads_mapping[first_thread_id][node_id];
  CHECK_LT(mapped_nod_id, (*histograms)[first_thread_id].size());
  GradientSumT* hist0 =  (*histograms)[first_thread_id][mapped_nod_id].data();

  for (size_t bin_id = begin; bin_id < end; ++bin_id) {
    dest_hist[bin_id] = hist0[bin_id];
    hist0[bin_id] = 0;
  }
  for (size_t tid = 1; tid < threads_id_for_node.size(); ++tid) {
    const size_t thread_id = threads_id_for_node[tid];
    const size_t mapped_nod_id = local_threads_mapping[thread_id][node_id];
    GradientSumT* hist =  (*histograms)[thread_id][mapped_nod_id].data();
    for (size_t bin_id = begin; bin_id < end; ++bin_id) {
      dest_hist[bin_id] += hist[bin_id];
      hist[bin_id] = 0;
    }
  }
}

template void ReduceHist(float* dest_hist,
                         const std::vector<std::vector<uint16_t>>& local_threads_mapping,
                         std::vector<std::vector<std::vector<float>>>* histograms,
                         const size_t node_displace,
                         const std::vector<uint16_t>& threads_id_for_node,
                         size_t begin, size_t end);
template void ReduceHist(double* dest_hist,
                         const std::vector<std::vector<uint16_t>>& local_threads_mapping,
                         std::vector<std::vector<std::vector<double>>>* histograms,
                         const size_t node_displace,
                         const std::vector<uint16_t>& threads_id_for_node,
                         size_t begin, size_t end);

}  // namespace common
}  // namespace xgboost
