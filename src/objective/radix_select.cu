/**
 * Copyright 2026, XGBoost Contributors
 * \file radix_select.cu
 * \brief CUDA implementation of float32 radix selection.
 */
#include <dmlc/registry.h>  // for DMLC_REGISTRY_FILE_TAG

#include <cmath>    // fmax, fmin
#include <cstddef>  // std::size_t
#include <cstdint>  // std::uint32_t

#include "../collective/aggregator.h"    // for GlobalSum
#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for LaunchN
#include "../common/kernel.h"            // for KernelRegistration
#include "../common/optional_weight.h"   // for OptionalWeights
#include "radix_select.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(radix_select_kernel_cuda);

namespace {
constexpr std::size_t kRadixBits{8};
constexpr std::size_t kRadixBins{std::size_t{1} << kRadixBits};
constexpr std::size_t kRadixPasses{sizeof(float) * 8 / kRadixBits};

void RadixSelectCuda(Context const* ctx, linalg::Matrix<float> const& values,
                     HostDeviceVector<float> const& weights, HostDeviceVector<float> const& alphas,
                     bst_target_t n_targets, linalg::Vector<float>* out) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  values.SetDevice(device);
  weights.SetDevice(device);
  alphas.SetDevice(device);
  auto d_values = values.View(device);
  auto d_weights = common::OptionalWeights{weights.ConstDeviceSpan()};
  auto d_alphas = alphas.ConstDeviceSpan();
  auto n_rows = values.Shape(0);
  auto n_alphas = alphas.Size();
  auto n_outputs = n_targets;
  *out = linalg::Zeros<float>(ctx, n_outputs);
  if (n_outputs == 0) {
    return;
  }

  linalg::Vector<std::uint32_t> prefixes = linalg::Zeros<std::uint32_t>(ctx, n_outputs);
  linalg::Vector<double> ranks = linalg::Zeros<double>(ctx, n_outputs);
  linalg::Vector<double> histogram = linalg::Zeros<double>(ctx, n_outputs * kRadixBins);
  auto d_prefixes = prefixes.View(device);
  auto d_ranks = ranks.View(device);
  auto d_histogram = histogram.View(device);
  auto stream = ctx->CUDACtx()->Stream();

  for (std::size_t pass{0}; pass < kRadixPasses; ++pass) {
    dh::LaunchN(d_histogram.Size(), stream,
                [=] __device__(std::size_t i) mutable { d_histogram(i) = 0.0; });
    auto shift = 32 - (pass + 1) * kRadixBits;
    auto prefix_mask = pass == 0 ? 0U : 0xffffffffU << (shift + 8);
    // Histogram the next byte of the order-preserving float key for matching prefixes. Atomic
    // updates replace the per-thread histograms used by the CPU implementation.
    dh::LaunchN(n_rows * n_outputs, stream, [=] __device__(std::size_t i) mutable {
      auto output = i / n_rows;
      auto row = i % n_rows;
      auto column = output / n_alphas;
      auto bits = __float_as_uint(d_values(row, column));
      // Complement negatives and flip the sign bit of non-negatives so unsigned key order agrees
      // with numeric float order. This is the device equivalent of ToOrderedKey in the CPU file.
      auto key = bits ^ (bits & 0x80000000U ? 0xffffffffU : 0x80000000U);
      if ((key & prefix_mask) != d_prefixes(output)) {
        return;
      }
      auto bin = (key >> shift) & (kRadixBins - 1);
      atomicAdd(&d_histogram(output * kRadixBins + bin), static_cast<double>(d_weights[row]));
    });
    // All workers must choose the same prefix, so select from the globally summed histogram.
    collective::SafeColl(collective::GlobalSum(ctx, d_histogram));

    dh::LaunchN(n_outputs, stream, [=] __device__(std::size_t output) mutable {
      auto alpha = d_alphas[output % n_alphas];
      if (pass == 0) {
        double total{0.0};
        for (std::size_t bin{0}; bin < kRadixBins; ++bin) {
          total += d_histogram(output * kRadixBins + bin);
        }
        if (total == 0.0) {
          d_ranks(output) = -1.0;
          return;
        }
        d_ranks(output) = alpha * total;
      } else if (d_ranks(output) < 0.0) {
        return;
      }

      std::size_t selected{0};
      double before{0.0};
      if (alpha == 0.0f) {
        for (std::size_t bin{0}; bin < kRadixBins; ++bin) {
          if (d_histogram(output * kRadixBins + bin) > 0.0) {
            selected = bin;
            break;
          }
        }
      } else if (alpha == 1.0f) {
        for (std::size_t bin = kRadixBins; bin-- > 0;) {
          if (d_histogram(output * kRadixBins + bin) > 0.0) {
            selected = bin;
            break;
          }
        }
        for (std::size_t bin{0}; bin < selected; ++bin) {
          before += d_histogram(output * kRadixBins + bin);
        }
      } else {
        double cumulative{0.0};
        for (std::size_t bin{0}; bin < kRadixBins; ++bin) {
          auto weight = d_histogram(output * kRadixBins + bin);
          cumulative += weight;
          // Keep the last non-empty bin as a fallback in case rounding leaves the remaining rank
          // just above the accumulated weight.
          if (weight > 0.0) {
            selected = bin;
            before = cumulative - weight;
          }
          if (weight > 0.0 && cumulative >= d_ranks(output)) {
            break;
          }
        }
      }
      // Retain the rank within the selected bin and append that bin as the next prefix byte.
      auto selected_weight = d_histogram(output * kRadixBins + selected);
      d_ranks(output) = fmin(fmax(d_ranks(output) - before, 0.0), selected_weight);
      d_prefixes(output) |= static_cast<std::uint32_t>(selected) << shift;
    });
  }

  // Reverse the ordered-key transform after all four prefix bytes have been selected.
  auto d_out = out->View(device);
  dh::LaunchN(n_outputs, stream, [=] __device__(std::size_t output) mutable {
    auto key = d_prefixes(output);
    auto bits = key ^ (key & 0x80000000U ? 0x80000000U : 0xffffffffU);
    d_out(output) = d_ranks(output) < 0.0 ? 0.0f : __uint_as_float(bits);
  });
}

auto const kRegisterRadixSelectCuda =
    common::KernelRegistration<RadixSelectKernel>{DeviceOrd::kCUDA, &RadixSelectCuda};
}  // namespace
}  // namespace xgboost::obj
