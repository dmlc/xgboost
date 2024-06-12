/**
 * Copyright 2024, XGBoost contributors
 *
 * @brief This file defines the interface required for a federated plugin to implement.

 * For federated learning, operations for the gradient and gradient histogram are
 * performed in the encrypted space, and the encryption is provided by a third-party
 * plugin. The interface is split into four sections:
 *
 *   - Library handle.
 *   - Gradient encryption.
 *   - Build histogram for vertical federated learning.
 *   - Build histogram for horizontal federated learning.
 *
 * See below function prototypes for details. All prototypes are for C functions that are
 * suitable for `dlopen`.
 */
#pragma once

#include <algorithm>   // for copy_n
#include <cstdint>     // for uint8_t
#include <functional>  // for function
#include <memory>      // for unique_ptr
#include <vector>      // for vector

#include "xgboost/json.h"         // for Json
#include "xgboost/span.h"         // for Span
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::collective {
namespace federated {
/**
 * @brief Functions for the plugin handle. Plugin can use an opaque handle for defining
 *        private data structures.
 */
typedef void *FederatedPluginHandle;  // NOLINT

using CreateFn = FederatedPluginHandle(int, char const **);
using CloseFn = int(FederatedPluginHandle);
using ErrorFn = char const *();
/**
 * @brief Gradient functions, used to provide encryption for gradients.
 */
using EncryptFn = int(FederatedPluginHandle handle, float const *in_gpair, size_t n_in,
                      uint8_t **out_gpair, size_t *n_out);
using SyncEncryptFn = int(FederatedPluginHandle handle, uint8_t const *in_gpair, size_t n_bytes,
                          uint8_t **out_gpair, size_t *n_out);
/**
 * @brief Vertical federated learning histogram functions.
 */
using ResetHistCtxVertFn = int(FederatedPluginHandle handle, uint32_t const *cutptrs,
                               size_t cutptr_len, int32_t const *bin_idx, size_t n_idx);
using BuildHistVertFn = int(FederatedPluginHandle handle, uint64_t const **ridx,
                            size_t const *sizes, int32_t const *nidx, size_t len,
                            uint8_t **out_hist, size_t *out_len);
using SyncHistVertFn = int(FederatedPluginHandle handle, uint8_t *buf, size_t len, double **out,
                           size_t *out_len);
/**
 * @brief Horizontal federated learning histogram functions.
 */
using BuildHistHoriFn = int(FederatedPluginHandle handle, double const *in_histogram, size_t len,
                            uint8_t **out_hist, size_t *out_len);
using SyncHistHoriFn = int(FederatedPluginHandle handle, uint8_t const *buffer, size_t len,
                           double **out_hist, size_t *out_len);
}  // namespace federated

// Base class for federated learning plugin.
class FederatedPluginBase {
  std::vector<std::uint8_t> grad_;
  std::vector<std::uint8_t> hist_enc_;
  std::vector<double> hist_plain_;

 public:
  [[nodiscard]] virtual common::Span<std::uint8_t> EncryptGradient(
      common::Span<float const> data) = 0;
  virtual void SyncEncryptedGradient(common::Span<std::uint8_t const>) = 0;

  // Vertical histogram
  virtual void Reset(common::Span<std::uint32_t const>, common::Span<std::int32_t const>) {}

  [[nodiscard]] virtual common::Span<std::uint8_t> BuildEncryptedHistVert(
      common::Span<std::uint64_t const *> rowptrs, common::Span<std::size_t const> sizes,
      common::Span<bst_node_t const> nids) = 0;

  [[nodiscard]] virtual common::Span<double> SyncEncryptedHistVert(
      common::Span<std::uint8_t> hist) = 0;

  // Horizontal histogram
  [[nodiscard]] virtual common::Span<std::uint8_t> BuildEncryptedHistHori(
      common::Span<double const> hist) = 0;
  [[nodiscard]] virtual common::Span<double> SyncEncryptedHistHori(
      common::Span<std::uint8_t const> hist) = 0;
};

// Only used for testing, this class is an no-op implementation.
class FederatedPluginMock : public FederatedPluginBase {
  std::vector<std::uint8_t> grad_;

  std::vector<std::uint8_t> hist_enc_;  // represents the encrypted histogram
  std::vector<double> hist_plain_;      // represents the plain text histogram

  std::vector<std::uint32_t> cuts_;  // HistogramCuts::Ptrs()
  std::vector<bst_bin_t> gidx_;      // GHistIndexMatrix

 public:
  ~FederatedPluginMock() = default;

  [[nodiscard]] common::Span<std::uint8_t> EncryptGradient(
      common::Span<float const> data) override {
    grad_.resize(data.size_bytes());
    auto casted =
        common::Span{reinterpret_cast<std::uint8_t const *>(data.data()), data.size_bytes()};
    std::copy_n(casted.data(), casted.size(), grad_.data());
    return grad_;
  }
  void SyncEncryptedGradient(common::Span<std::uint8_t const> data) override {
    auto casted =
        common::Span{reinterpret_cast<float const *>(data.data()), data.size() / sizeof(float)};
    grad_.resize(casted.size());
    std::copy_n(casted.data(), casted.size(), grad_.data());
  }

  // Vertical histogram
  void Reset(common::Span<std::uint32_t const>, common::Span<std::int32_t const>) override;

  [[nodiscard]] common::Span<std::uint8_t> BuildEncryptedHistVert(
      common::Span<std::uint64_t const *> rowptrs, common::Span<std::size_t const> sizes,
      common::Span<bst_node_t const> nids) override;

  [[nodiscard]] common::Span<double> SyncEncryptedHistVert(
      common::Span<std::uint8_t> hist) override;

  // Horizontal histogram
  [[nodiscard]] common::Span<std::uint8_t> BuildEncryptedHistHori(
      common::Span<double const> hist) override {
    hist_enc_.resize(hist.size_bytes());
    std::copy_n(reinterpret_cast<std::uint8_t const *>(hist.data()), hist.size_bytes(),
                hist_enc_.data());
    return hist_enc_;
  }
  [[nodiscard]] common::Span<double> SyncEncryptedHistHori(
      common::Span<std::uint8_t const> hist) override {
    std::size_t n = hist.size_bytes() / sizeof(double);
    hist_plain_.resize(n);
    std::copy_n(reinterpret_cast<double const *>(hist.data()), n, hist_plain_.data());
    return hist_plain_;
  }
};

/**
 * @brief Bridge for plugins that handle encryption.
 */
class FederatedPlugin : public FederatedPluginBase {
  // Federated plugin shared object, for dlopen.
  std::unique_ptr<void, std::function<void(void *)>> plugin_;

  federated::CreateFn *PluginCreate_{nullptr};
  federated::CloseFn *PluginClose_{nullptr};
  federated::ErrorFn *ErrorMsg_{nullptr};
  // Gradient
  federated::EncryptFn *Encrypt_{nullptr};
  federated::SyncEncryptFn *SyncEncrypt_{nullptr};
  // Vert Histogram
  federated::ResetHistCtxVertFn *ResetHistCtxVert_{nullptr};
  federated::BuildHistVertFn *BuildEncryptedHistVert_{nullptr};
  federated::SyncHistVertFn *SyncEncryptedHistVert_{nullptr};
  // Hori Histogram
  federated::BuildHistHoriFn *BuildEncryptedHistHori_{nullptr};
  federated::SyncHistHoriFn *SyncEncryptedHistHori_;

  // Object handle of the plugin.
  std::unique_ptr<void, std::function<void(void *)>> plugin_handle_;

  void CheckRC(std::int32_t rc, std::string_view msg) {
    if (rc != 0) {
      auto err_msg = ErrorMsg_();
      LOG(FATAL) << msg << ":" << err_msg;
    }
  }

 public:
  explicit FederatedPlugin(StringView path, Json config);
  ~FederatedPlugin();
  // Gradient
  [[nodiscard]] common::Span<std::uint8_t> EncryptGradient(
      common::Span<float const> data) override {
    std::uint8_t *ptr{nullptr};
    std::size_t n{0};
    auto rc = Encrypt_(this->plugin_handle_.get(), data.data(), data.size(), &ptr, &n);
    CheckRC(rc, "Failed to encrypt gradient");
    return {ptr, n};
  }
  void SyncEncryptedGradient(common::Span<std::uint8_t const> data) override {
    uint8_t *out;
    std::size_t n{0};
    auto rc = SyncEncrypt_(this->plugin_handle_.get(), data.data(), data.size(), &out, &n);
    CheckRC(rc, "Failed to sync encrypt gradient");
  }

  // Vertical histogram
  void Reset(common::Span<std::uint32_t const> cutptrs,
             common::Span<std::int32_t const> bin_idx) override {
    auto rc = ResetHistCtxVert_(this->plugin_handle_.get(), cutptrs.data(), cutptrs.size(),
                                bin_idx.data(), bin_idx.size());
    CheckRC(rc, "Failed to set the data context for federated learning");
  }
  [[nodiscard]] common::Span<std::uint8_t> BuildEncryptedHistVert(
      common::Span<std::uint64_t const *> rowptrs, common::Span<std::size_t const> sizes,
      common::Span<bst_node_t const> nids) override {
    std::uint8_t *ptr{nullptr};
    std::size_t n{0};
    auto rc = BuildEncryptedHistVert_(this->plugin_handle_.get(), rowptrs.data(), sizes.data(),
                                      nids.data(), nids.size(), &ptr, &n);
    CheckRC(rc, "Failed to build the encrypted hist");
    return {ptr, n};
  }
  [[nodiscard]] common::Span<double> SyncEncryptedHistVert(
      common::Span<std::uint8_t> hist) override {
    double *ptr{nullptr};
    std::size_t n{0};
    auto rc =
        SyncEncryptedHistVert_(this->plugin_handle_.get(), hist.data(), hist.size(), &ptr, &n);
    CheckRC(rc, "Failed to sync the encrypted hist");
    return {ptr, n};
  }

  // Horizontal histogram
  [[nodiscard]] common::Span<std::uint8_t> BuildEncryptedHistHori(
      common::Span<double const> hist) override {
    std::uint8_t *ptr{nullptr};
    std::size_t n{0};
    auto rc =
        BuildEncryptedHistHori_(this->plugin_handle_.get(), hist.data(), hist.size(), &ptr, &n);
    CheckRC(rc, "Failed to build the encrypted hist");
    return {ptr, n};
  }
  [[nodiscard]] common::Span<double> SyncEncryptedHistHori(
      common::Span<std::uint8_t const> hist) override {
    double *ptr{nullptr};
    std::size_t n{0};
    auto rc =
        SyncEncryptedHistHori_(this->plugin_handle_.get(), hist.data(), hist.size(), &ptr, &n);
    CheckRC(rc, "Failed to sync the encrypted hist");
    return {ptr, n};
  }
};

[[nodiscard]] FederatedPluginBase *CreateFederatedPlugin(Json config);
}  // namespace xgboost::collective
