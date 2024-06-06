/**
 * Copyright 2024, XGBoost contributors
 */
#pragma once

#include <functional>   // for function
#include <memory>       // for unique_ptr
#include <string_view>  // for string_view

#include "xgboost/json.h"  // for Json
#include "xgboost/span.h"  // for Span

typedef void *FederatedPluginHandle;  // NOLINT

namespace xgboost::collective {
namespace federated {
// API exposed by the plugin
using CreateFn = FederatedPluginHandle(int, char const **);
using CloseFn = int(FederatedPluginHandle);
using ErrorFn = char const *();
// Gradient
using EncryptFn = int(FederatedPluginHandle handle, float const *in_gpair, size_t n_in,
                      uint8_t **out_gpair, size_t *n_out);
using SyncEncryptFn = int(FederatedPluginHandle handle, uint8_t const *in_gpair, size_t n_bytes,
                          uint8_t const **out_gpair, size_t *n_out);
// Vert Histogram
using ResetHistCtxVertFn = int(FederatedPluginHandle handle, std::uint32_t const *cutptrs,
                               std::size_t cutptr_len, std::int32_t const *bin_idx,
                               std::size_t n_idx);
using BuildHistVertFn = int(FederatedPluginHandle handle, uint64_t const **ridx,
                            size_t const *sizes, int32_t const *nidx, size_t len,
                            uint8_t **out_hist, size_t *out_len);
using SyncHistVertFn = int(FederatedPluginHandle handle, uint8_t *buf, size_t len, double **out,
                           size_t *out_len);
// Hori Histogram
using BuildHistHoriFn = int(FederatedPluginHandle handle, double const *in_histogram, size_t len,
                            uint8_t **out_hist, size_t *out_len);
using SyncHistHoriFn = int(FederatedPluginHandle handle, std::uint8_t const *buffer,
                           std::size_t len, double **out_hist, std::size_t *out_len);
}  // namespace federated

/**
 * @brief Bridge for plugins that handle encryption.
 */
class FederatedPlugin {
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
  explicit FederatedPlugin(std::string_view path, Json config);
  ~FederatedPlugin();
  // Gradient
  [[nodiscard]] virtual common::Span<std::uint8_t> EncryptGradient(common::Span<float const> data) {
    std::uint8_t *ptr{nullptr};
    std::size_t n{0};
    auto rc = Encrypt_(this->plugin_handle_.get(), data.data(), data.size(), &ptr, &n);
    CheckRC(rc, "Failed to encrypt gradient");
    return {ptr, n};
  }
  virtual void SyncEncryptedGradient(common::Span<std::uint8_t const> data) {
    uint8_t const *out;
    std::size_t n{0};
    auto rc = SyncEncrypt_(this->plugin_handle_.get(), data.data(), data.size(), &out, &n);
    CheckRC(rc, "Failed to sync encrypt gradient");
  }

  // Vertical histogram
  virtual void Reset(common::Span<std::uint32_t const> cutptrs,
                     common::Span<std::int32_t const> bin_idx) {
    auto rc = ResetHistCtxVert_(this->plugin_handle_.get(), cutptrs.data(), cutptrs.size(),
                                bin_idx.data(), bin_idx.size());
    CheckRC(rc, "Failed to set the data context for federated learning");
  }
  [[nodiscard]] virtual common::Span<std::uint8_t> BuildEncryptedHistVert(
      common::Span<std::uint64_t const *> rowptrs, common::Span<std::size_t const> sizes,
      common::Span<bst_node_t const> nids) {
    std::uint8_t *ptr{nullptr};
    std::size_t n{0};
    auto rc = BuildEncryptedHistVert_(this->plugin_handle_.get(), rowptrs.data(), sizes.data(),
                                      nids.data(), nids.size(), &ptr, &n);
    CheckRC(rc, "Failed to build the encrypted hist");
    return {ptr, n};
  }
  [[nodiscard]] virtual common::Span<double> SyncEncryptedHistVert(
      common::Span<std::uint8_t> hist) {
    double *ptr{nullptr};
    std::size_t n{0};
    auto rc =
        SyncEncryptedHistVert_(this->plugin_handle_.get(), hist.data(), hist.size(), &ptr, &n);
    CheckRC(rc, "Failed to sync the encrypted hist");
    return {ptr, n};
  }

  // Horizontal histogram
  [[nodiscard]] virtual common::Span<std::uint8_t> BuildEncryptedHistHori(
      common::Span<double const> hist) {
    std::uint8_t *ptr{nullptr};
    std::size_t n{0};
    auto rc =
        BuildEncryptedHistHori_(this->plugin_handle_.get(), hist.data(), hist.size(), &ptr, &n);
    CheckRC(rc, "Failed to build the encrypted hist");
    return {ptr, n};
  }
  [[nodiscard]] virtual common::Span<double> SyncEncryptedHistHori(
      common::Span<std::uint8_t const> hist) {
    double *ptr{nullptr};
    std::size_t n{0};
    auto rc =
        SyncEncryptedHistHori_(this->plugin_handle_.get(), hist.data(), hist.size(), &ptr, &n);
    CheckRC(rc, "Failed to sync the encrypted hist");
    return {ptr, n};
  }
};

class FederatedPluginMock : public FederatedPlugin {
 public:
};
}  // namespace xgboost::collective
