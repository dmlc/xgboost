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
 * Since we don't require the plugin to have network capability, the synchronization is
 * performed in XGBoost. As a result, the build procedure is divided into four steps,
 * first we need to build a local histogram, then encrypt it with the plugin. Afterward,
 * the control returns to XBGoost, which is responsible for synchronization. Lastly, the
 * plugin will recieve the synchronization result and return the decrypted histogram.
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

#include "../../src/data/gradient_index.h"  // for GHistIndexMatrix
#include "xgboost/json.h"                   // for Json
#include "xgboost/span.h"                   // for Span
#include "xgboost/string_view.h"            // for StringView

namespace xgboost::collective {
namespace federated {
/**
 * @defgroup Functions for the plugin handle.
 *
 * Plugin can use an opaque handle for defining private data structures and needed
 * context.
 */
typedef void *FederatedPluginHandle;  // NOLINT
/**
 * @brief Create a handle for the plugin.
 *
 *  Symbol name: `FederatedPluginCreate`.
 *
 * @return Returns nullptr if failed.
 */
using CreateFn = FederatedPluginHandle(int, char const **);
/**
 * @brief Close the handle after use.
 *
 *  Symbol name: `FederatedPluginClose`.
 *
 * @return 0 if succees.
 */
using CloseFn = int(FederatedPluginHandle);
/**
 * @brief Report error, if there's any.
 *
 *  Symbol name: `FederatedPluginErrorMsg`.
 */
using ErrorFn = char const *();
/**@}*/

/**
 * @defgroup Gradient functions
 *
 * Used to provide encryption for gradients.
 *
 * @{
 */
/**
 * @brief Encrypt the gradient on the active party.
 *
 *  Symbol name: `FederatedPluginEncryptGPairs`.
 *
 * @return 0 if succees.
 */
using EncryptFn = int(FederatedPluginHandle handle, float const *in_gpair, size_t n_in,
                      uint8_t **out_gpair, size_t *n_out);
/**
 * @brief Store the gradient for all parties after broadcast.
 *
 *  Symbol name: `FederatedPluginSyncEncryptedGPairs`.
 *
 * @return 0 if succees.
 */
using SyncEncryptFn = int(FederatedPluginHandle handle, uint8_t const *in_gpair, size_t n_bytes,
                          uint8_t **out_gpair, size_t *n_out);
/**@}*/

/**
 * @defgroup Vertical federated learning histogram functions.
 */
/**
 * @brief Set the context and data for building vertical histogram.
 *
 * For now, this assumes relatively dense input and copies the histogram bin index as a
 * dense matrix. In the future, we can optimize for sparse matrix if the need comes up.
 *
 *  Symbol name: `FederatedPluginResetHistContextVert`.
 *
 * @param cutptrs CSC pointers of the histogram cut matrix.
 * @param cutptr_len The number of the CSC pointers (n_features + 1).
 * @param bin_idx Gradient index of the histogram.
 * @param n_idx The number of indices. Equals to the size of the dataset, stored in row-major.
 *
 * @return 0 if succees.
 */
using ResetHistCtxVertFn = int(FederatedPluginHandle handle, uint32_t const *cutptrs,
                               size_t cutptr_len, int32_t const *bin_idx, size_t n_idx);
/**
 * @brief Build local encrypted histogram for vertical learning.
 *
 *  Symbol name: `FederatedPluginBuildEncryptedHistVert`.
 *
 * @param ridx  Row indices for each tree leaf.
 * @param sizes The number of rows for each tree leaf.
 * @param nidx  The node index of each tree leaf.
 * @param len   The number of leaves.
 * @param out_hist Output histogram.
 * @param out_len  The size of the output histogram.
 *
 * @return 0 if succees.
 */
using BuildHistVertFn = int(FederatedPluginHandle handle, uint64_t const **ridx,
                            size_t const *sizes, int32_t const *nidx, size_t len,
                            uint8_t **out_hist, size_t *out_len);
/**
 * @brief Synchronize the histogram after the allgather call for all parties.
 *
 *  Symbol name: `FederatedPluginSyncEncryptedHistVert`.
 *
 * @param in_hist Histogram buffer from the allgather call.
 * @param len     The size of the input histogram buffer.
 * @param out     Reduced histogram.
 * @param out_len The size of the reduced histogram.
 *
 * @return 0 if succees.
 */
using SyncHistVertFn = int(FederatedPluginHandle handle, uint8_t *in_hist, size_t len,
                           double **out_hist, size_t *out_len);
/**@}*/

/**
 * @defgroup Horizontal federated learning histogram functions.
 */
/**
 * @brief Encrypt the input histogram.
 *
 * The local histogram is built by XGBoost.
 *
 *  Symbol name `FederatedPluginBuildEncryptedHistHori`.
 *
 * @param in_hist  The input local histogram.
 * @param len      The size of the input local histogram.
 * @param out_hist Encrypted histogram.
 * @param out_len  The size of the encrypted histogram.
 *
 * @return 0 if succees.
 */
using BuildHistHoriFn = int(FederatedPluginHandle handle, double const *in_hist, size_t len,
                            uint8_t **out_hist, size_t *out_len);
/**
 * @brief Reduce the histogram after the allgather call.
 *
 *  Symbol name: `FederatedPluginSyncEncryptedHistHori`.
 *
 * @param in_hist  Input histogram from the allgather call.
 * @param len      The length of the input histogram.
 * @param out_hist Output histogram.
 * @param out_len  The size of the output histogram.
 *
 * @return 0 if succees.
 */
using SyncHistHoriFn = int(FederatedPluginHandle handle, uint8_t const *in_hist, size_t len,
                           double **out_hist, size_t *out_len);
/**@}*/
}  // namespace federated

// Base class for federated learning plugin.
class FederatedPluginBase {
  std::vector<std::uint8_t> grad_;
  std::vector<std::uint8_t> hist_enc_;
  std::vector<double> hist_plain_;

 public:
  virtual ~FederatedPluginBase() = default;

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
  Context ctx_;
  std::vector<std::uint8_t> grad_;

  std::vector<std::uint8_t> hist_enc_;  // represents the encrypted histogram
  std::vector<double> hist_plain_;      // represents the plain text histogram

  std::vector<std::uint32_t> cuts_;  // HistogramCuts::Ptrs()
  GHistIndexMatrix gmat_;

 public:
  ~FederatedPluginMock() override = default;

  [[nodiscard]] common::Span<std::uint8_t> EncryptGradient(
      common::Span<float const> data) override {
    grad_.resize(data.size_bytes());
    auto casted =
        common::Span{reinterpret_cast<std::uint8_t const *>(data.data()), data.size_bytes()};
    std::copy_n(casted.data(), casted.size(), grad_.data());
    return grad_;
  }
  void SyncEncryptedGradient(common::Span<std::uint8_t const> data) override {
    grad_.resize(data.size_bytes());
    std::copy_n(data.data(), data.size(), grad_.data());
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

  void CheckRC(std::int32_t rc, StringView msg) const {
    if (rc != 0) {
      auto err_msg = ErrorMsg_();
      LOG(FATAL) << msg << ":" << err_msg;
    }
  }

 public:
  explicit FederatedPlugin(StringView path, Json config);
  ~FederatedPlugin() override;
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

[[nodiscard]] std::shared_ptr<FederatedPluginBase> CreateFederatedPlugin(Json config);
}  // namespace xgboost::collective
