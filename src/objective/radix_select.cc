/**
 * Copyright 2026, XGBoost Contributors
 * \file radix_select.cc
 * \brief CPU implementation of float32 radix selection.
 */
#include "radix_select.h"

#include <algorithm>  // std::clamp, std::fill, std::max
#include <cstddef>    // std::size_t
#include <cstdint>    // std::uint32_t
#include <cstring>    // std::memcpy
#include <limits>     // std::numeric_limits
#include <numeric>    // std::accumulate
#include <vector>     // std::vector

#include "../collective/aggregator.h"   // for GlobalSum
#include "../common/kernel.h"           // for DispatchKernel, KernelRegistration
#include "../common/optional_weight.h"  // for OptionalWeights
#include "../common/threading_utils.h"  // for ParallelFor
#include "xgboost/logging.h"            // for CHECK

namespace xgboost::obj {
namespace {
constexpr std::size_t kRadixBits{8};
constexpr std::size_t kRadixBins{std::size_t{1} << kRadixBits};
constexpr std::size_t kRadixPasses{sizeof(float) * 8 / kRadixBits};

static_assert(sizeof(float) == sizeof(std::uint32_t));

// Unsigned comparison of raw IEEE-754 bits does not match numeric comparison: the sign bit puts
// negative values after positive values, and the magnitude bits run backwards within the negative
// range. Complementing every bit of a negative value and flipping only the sign bit of a
// non-negative value produces an unsigned key whose order matches the numeric float order.
std::uint32_t ToOrderedKey(float value) {
  std::uint32_t bits{0};
  std::memcpy(&bits, &value, sizeof(value));
  auto mask = bits & 0x80000000U ? std::numeric_limits<std::uint32_t>::max() : 0x80000000U;
  return bits ^ mask;
}

// Reverse the order-preserving transformation in ToOrderedKey.
float FromOrderedKey(std::uint32_t key) {
  auto mask = key & 0x80000000U ? 0x80000000U : std::numeric_limits<std::uint32_t>::max();
  auto bits = key ^ mask;
  float value{0.0f};
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Interior quantiles select the first bin whose cumulative weight reaches the rank. At alpha=0
// and alpha=1 the rank lies on a boundary, so explicitly choose the first or last non-empty bin.
std::size_t SelectBin(common::Span<double const> histogram, double rank, float alpha) {
  if (alpha == 0.0f) {
    for (std::size_t bin{0}; bin < histogram.size(); ++bin) {
      if (histogram[bin] > 0.0) {
        return bin;
      }
    }
  } else if (alpha == 1.0f) {
    for (std::size_t bin = histogram.size(); bin-- > 0;) {
      if (histogram[bin] > 0.0) {
        return bin;
      }
    }
  } else {
    double cumulative{0.0};
    for (std::size_t bin{0}; bin < histogram.size(); ++bin) {
      cumulative += histogram[bin];
      if (histogram[bin] > 0.0 && cumulative >= rank) {
        return bin;
      }
    }
  }
  // Rounding can leave rank just above cumulative weight. Prefer the last value-bearing bin over
  // returning an empty bin that cannot correspond to an input value.
  for (std::size_t bin = histogram.size(); bin-- > 0;) {
    if (histogram[bin] > 0.0) {
      return bin;
    }
  }
  return 0;
}

void RadixSelectCpu(Context const* ctx, linalg::Matrix<float> const& values,
                    HostDeviceVector<float> const& weights, HostDeviceVector<float> const& alphas,
                    bst_target_t n_targets, linalg::Vector<float>* out) {
  CHECK(ctx->IsCPU());
  auto n_alphas = alphas.Size();
  auto n_outputs = n_targets;
  *out = linalg::Zeros<float>(ctx, n_outputs);
  if (n_outputs == 0) {
    return;
  }

  auto h_values = values.HostView();
  auto h_weights = common::OptionalWeights{weights.ConstHostSpan()};
  auto h_alphas = alphas.ConstHostSpan();
  auto n_threads = std::max<std::int32_t>(ctx->Threads(), 1);
  // For each (column, alpha), prefix is the part of the selected key already known and rank is the
  // cumulative weight still required within that prefix.
  std::vector<std::uint32_t> prefixes(n_outputs, 0);
  std::vector<double> ranks(n_outputs, 0.0);
  std::vector<double> histogram(n_outputs * kRadixBins, 0.0);
  // A histogram per thread avoids synchronization while scanning the rows.
  std::vector<double> thread_histogram(n_threads * histogram.size(), 0.0);

  for (std::size_t pass{0}; pass < kRadixPasses; ++pass) {
    std::fill(thread_histogram.begin(), thread_histogram.end(), 0.0);
    auto shift = 32 - (pass + 1) * kRadixBits;
    auto prefix_mask = pass == 0 ? 0U : std::numeric_limits<std::uint32_t>::max() << (shift + 8);
    // Count the next byte only for keys matching the prefix chosen in earlier passes.
    common::ParallelFor(values.Shape(0) * n_outputs, n_threads, [&](std::size_t i) {
      auto output = i / values.Shape(0);
      auto row = i % values.Shape(0);
      auto column = output / n_alphas;
      auto key = ToOrderedKey(h_values(row, column));
      if ((key & prefix_mask) != prefixes[output]) {
        return;
      }
      auto bin = (key >> shift) & (kRadixBins - 1);
      auto thread = omp_get_thread_num();
      auto offset = (thread * n_outputs + output) * kRadixBins + bin;
      thread_histogram[offset] += h_weights[row];
    });

    // Combine thread-local histograms, then combine the same bins across workers.
    std::fill(histogram.begin(), histogram.end(), 0.0);
    for (std::int32_t thread{0}; thread < n_threads; ++thread) {
      auto offset = thread * histogram.size();
      for (std::size_t i{0}; i < histogram.size(); ++i) {
        histogram[i] += thread_histogram[offset + i];
      }
    }
    collective::SafeColl(
        collective::GlobalSum(ctx, linalg::MakeVec(histogram.data(), histogram.size())));

    for (std::size_t output{0}; output < n_outputs; ++output) {
      auto bins = common::Span<double const>{histogram.data() + output * kRadixBins, kRadixBins};
      auto alpha = h_alphas[output % n_alphas];
      if (pass == 0) {
        auto total = std::accumulate(bins.begin(), bins.end(), 0.0);
        if (total == 0.0) {
          ranks[output] = -1.0;
          continue;
        }
        ranks[output] = alpha * total;
      } else if (ranks[output] < 0.0) {
        continue;
      }
      // Keep the bin containing the remaining weighted rank. The selected bin becomes the next
      // prefix byte, while weights in preceding bins are removed from the rank for the next pass.
      auto bin = SelectBin(bins, ranks[output], alpha);
      auto weight_before = std::accumulate(bins.begin(), bins.begin() + bin, 0.0);
      ranks[output] = std::clamp(ranks[output] - weight_before, 0.0, bins[bin]);
      prefixes[output] |= static_cast<std::uint32_t>(bin) << shift;
    }
  }

  // All four bytes are now known, so the prefix is the ordered key of the selected float.
  auto h_out = out->HostView();
  for (std::size_t output{0}; output < n_outputs; ++output) {
    h_out(output) = ranks[output] < 0.0 ? 0.0f : FromOrderedKey(prefixes[output]);
  }
}

auto const kRegisterRadixSelectCpu =
    common::KernelRegistration<RadixSelectKernel>{DeviceOrd::kCPU, &RadixSelectCpu};
}  // namespace

void RadixSelect(Context const* ctx, linalg::Matrix<float> const& values,
                 HostDeviceVector<float> const& weights, HostDeviceVector<float> const& alphas,
                 bst_target_t n_targets, linalg::Vector<float>* out) {
  CHECK(ctx);
  CHECK(weights.Empty() || weights.Size() == values.Shape(0));
  CHECK(!alphas.Empty());
  CHECK_NE(n_targets, 0);
  CHECK(values.Shape(1) == 0 || values.Shape(1) * alphas.Size() == n_targets)
      << "Invalid number of outputs.";
  for (auto alpha : alphas.ConstHostSpan()) {
    CHECK_GE(alpha, 0.0f);
    CHECK_LE(alpha, 1.0f);
  }
  common::DispatchKernel<RadixSelectKernel>(ctx, values, weights, alphas, n_targets, out);
}
}  // namespace xgboost::obj
