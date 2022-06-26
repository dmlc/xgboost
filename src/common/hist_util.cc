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

HistogramCuts SketchOnDMatrix(DMatrix *m, int32_t max_bins, int32_t n_threads, bool use_sorted,
                              Span<float> const hessian) {
  HistogramCuts out;
  auto const& info = m->Info();
  std::vector<bst_row_t> reduced(info.num_col_, 0);
  for (auto const &page : m->GetBatches<SparsePage>()) {
    auto const &entries_per_column =
        CalcColumnSize(data::SparsePageAdapterBatch{page.GetView()}, info.num_col_, n_threads,
                       [](auto) { return true; });
    CHECK_EQ(entries_per_column.size(), info.num_col_);
    for (size_t i = 0; i < entries_per_column.size(); ++i) {
      reduced[i] += entries_per_column[i];
    }
  }

  if (!use_sorted) {
    HostSketchContainer container(max_bins, m->Info(), reduced, HostSketchContainer::UseGroup(info),
                                  n_threads);
    for (auto const& page : m->GetBatches<SparsePage>()) {
      container.PushRowPage(page, info, hessian);
    }
    container.MakeCuts(&out);
  } else {
    SortedSketchContainer container{max_bins, m->Info(), reduced,
                                    HostSketchContainer::UseGroup(info), n_threads};
    for (auto const& page : m->GetBatches<SortedCSCPage>()) {
      container.PushColPage(page, info, hessian);
    }
    container.MakeCuts(&out);
  }

  return out;
}

/*!
 * \brief fill a histogram by zeros in range [begin, end)
 */
void InitilizeHistByZeroes(GHistRow hist, size_t begin, size_t end) {
#if defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  std::fill(hist.begin() + begin, hist.begin() + end, xgboost::GradientPairPrecise());
#else  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  memset(hist.data() + begin, '\0', (end - begin) * sizeof(xgboost::GradientPairPrecise));
#endif  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
}

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
void IncrementHist(GHistRow dst, const GHistRow add, size_t begin, size_t end) {
  double* pdst = reinterpret_cast<double*>(dst.data());
  const double *padd = reinterpret_cast<const double *>(add.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] += padd[i];
  }
}

/*!
 * \brief Copy hist from src to dst in range [begin, end)
 */
void CopyHist(GHistRow dst, const GHistRow src, size_t begin, size_t end) {
  double *pdst = reinterpret_cast<double *>(dst.data());
  const double *psrc = reinterpret_cast<const double *>(src.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc[i];
  }
}

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
void SubtractionHist(GHistRow dst, const GHistRow src1, const GHistRow src2, size_t begin,
                     size_t end) {
  double* pdst = reinterpret_cast<double*>(dst.data());
  const double* psrc1 = reinterpret_cast<const double*>(src1.data());
  const double* psrc2 = reinterpret_cast<const double*>(src2.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc1[i] - psrc2[i];
  }
}

void ClearHist(double* dest_hist,
                size_t begin, size_t end) {
  for (size_t bin_id = begin; bin_id < end; ++bin_id) {
    dest_hist[bin_id]  = 0;
  }
}

void ReduceHist(double* dest_hist,
                const std::vector<std::vector<uint16_t>>& local_threads_mapping,
                std::vector<std::vector<std::vector<double>>>* histograms,
                const size_t node_id,
                const std::vector<uint16_t>& threads_id_for_node,
                size_t begin, size_t end) {
  const size_t first_thread_id = threads_id_for_node[0];
  CHECK_LT(node_id, local_threads_mapping[first_thread_id].size());
  const size_t mapped_nod_id = local_threads_mapping[first_thread_id][node_id];
  CHECK_LT(mapped_nod_id, (*histograms)[first_thread_id].size());
  double* hist0 =  (*histograms)[first_thread_id][mapped_nod_id].data();

  for (size_t bin_id = begin; bin_id < end; ++bin_id) {
    dest_hist[bin_id] = hist0[bin_id];
    hist0[bin_id] = 0;
  }
  for (size_t tid = 1; tid < threads_id_for_node.size(); ++tid) {
    const size_t thread_id = threads_id_for_node[tid];
    const size_t mapped_nod_id = local_threads_mapping[thread_id][node_id];
    double* hist =  (*histograms)[thread_id][mapped_nod_id].data();
    for (size_t bin_id = begin; bin_id < end; ++bin_id) {
      dest_hist[bin_id] += hist[bin_id];
      hist[bin_id] = 0;
    }
  }
}

}  // namespace common
}  // namespace xgboost
