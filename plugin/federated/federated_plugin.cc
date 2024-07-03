/**
 * Copyright 2024, XGBoost Contributors
 */
#include "federated_plugin.h"

#include <dlfcn.h>  // for dlclose, dlsym, dlopen

#include <cstddef>  // for size_t
#include <cstdint>  // for uint8_t, uint64_t
#include <sstream>  // for stringstream

#include "../../src/common/json_utils.h"  // for OptionalArg
#include "../../src/data/gradient_index.h"
#include "../../src/common/type.h"        // for RestoreType
#include "xgboost/base.h"                 // for bst_bin_t, bst_feature_t
#include "xgboost/json.h"                 // for Json
#include "xgboost/logging.h"              // for CHECK_EQ
#include "xgboost/span.h"                 // for Span
#include "xgboost/string_view.h"          // for StringView

namespace xgboost::collective {
void FederatedPluginMock::Reset(common::Span<std::uint32_t const> cutptrs,
                                common::Span<std::int32_t const> bin_idx) {
  this->cuts_.resize(cutptrs.size());
  std::copy_n(cutptrs.data(), cutptrs.size(), this->cuts_.data());

  // Restore the GHist index
  gmat_.max_numeric_bins_per_feat = 0;
  for (std::size_t i = 1; i < cuts_.size(); ++i) {
    auto begin = cuts_[i - 1];
    auto end = cuts_[i];
    gmat_.max_numeric_bins_per_feat =
        std::max(static_cast<bst_bin_t>(end - begin), gmat_.max_numeric_bins_per_feat);
  }
  auto is_valid = [](auto bin) {
    return bin >= 0;
  };
  std::size_t nnz = std::count_if(bin_idx.cbegin(), bin_idx.cend(), is_valid);
  gmat_.ResizeIndex(nnz, /*is_dense=*/nnz == bin_idx.size());
  gmat_.SetDense(nnz == bin_idx.size());
  common::DispatchBinType(gmat_.index.GetBinTypeSize(), [&](auto t) {
    auto data = gmat_.index.data<decltype(t)>();
    std::copy_if(bin_idx.cbegin(), bin_idx.cend(), data, is_valid);
  });

  bst_feature_t n_features = cuts_.size() - 1;
  bst_idx_t n_samples = bin_idx.size() / n_features;

  // For now, the gmat cannot be dense due to limiation of of encrypted training where
  // indices from other parties are marked as missing instead.
  gmat_.row_ptr = common::MakeFixedVecWithMalloc(n_samples + 1, std::size_t{0});
  auto gidx = linalg::MakeTensorView(&ctx_, bin_idx, n_samples, n_features);
  common::ParallelFor(n_samples, ctx_.Threads(), [&](auto i) {
    for (std::size_t j = 0; j < n_features; ++j) {
      if (is_valid(gidx(i, j))) {
        gmat_.row_ptr[i + 1]++;
      }
    }
  });
  std::partial_sum(gmat_.row_ptr.cbegin(), gmat_.row_ptr.cend(), gmat_.row_ptr.begin());
}

[[nodiscard]] common::Span<std::uint8_t> FederatedPluginMock::BuildEncryptedHistVert(
    common::Span<std::uint64_t const*> rowptrs, common::Span<std::size_t const> sizes,
    common::Span<bst_node_t const> nids) {
  bst_bin_t total_bin_size = cuts_.back();
  bst_idx_t n_samples = gmat_.Size();
  bst_bin_t hist_size = total_bin_size * 2;
  hist_plain_.resize(hist_size * nids.size());
  auto hist_buffer = common::Span<double>{hist_plain_};
  std::fill_n(hist_buffer.data(), hist_buffer.size(), 0.0);

  CHECK_EQ(rowptrs.size(), sizes.size());
  CHECK_EQ(nids.size(), sizes.size());
  auto gpair = common::RestoreType<GradientPair const>(common::Span<std::uint8_t>{grad_});
  CHECK_EQ(n_samples, gpair.size());

  common::ParallelFor(sizes.size(), ctx_.Threads(), [&](auto i) {
    auto hist_raw = hist_buffer.subspan(i * hist_size, hist_size);
    auto hist =
        common::Span{reinterpret_cast<GradientPairPrecise*>(hist_raw.data()), hist_raw.size() / 2};
    common::RowSetCollection::Elem row_indices{rowptrs[i], rowptrs[i] + sizes[i], nids[i]};
    if (gmat_.IsDense()) {
      common::BuildHist<false>(gpair, row_indices, gmat_, hist, false);
    } else {
      common::BuildHist<true>(gpair, row_indices, gmat_, hist, false);
    }
  });
  return {reinterpret_cast<std::uint8_t*>(hist_plain_.data()),
          common::Span<double>{hist_plain_}.size_bytes()};
}

[[nodiscard]] common::Span<double> FederatedPluginMock::SyncEncryptedHistVert(
    common::Span<std::uint8_t> hist) {
  hist_enc_.resize(hist.size());
  std::copy_n(hist.data(), hist.size(), hist_enc_.data());
  return common::RestoreType<double>(common::Span<std::uint8_t>{hist_enc_});
}

template <typename T>
auto SafeLoad(federated::FederatedPluginHandle handle, StringView name) {
  std::stringstream errs;
  auto ptr = reinterpret_cast<T>(dlsym(handle, name.c_str()));
  if (!ptr) {
    errs << "Failed to load symbol `" << name << "`. Error:\n  " << dlerror();
    LOG(FATAL) << errs.str();
  }
  return ptr;
}

FederatedPlugin::FederatedPlugin(StringView path, Json config)
    : plugin_{[&] {
                auto handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
                CHECK(handle) << "Failed to load federated plugin `" << path << "`:" << dlerror();
                return handle;
              }(),
              [](void* handle) {
                if (handle) {
                  auto rc = dlclose(handle);
                  if (rc != 0) {
                    LOG(WARNING) << "Failed to close federated plugin handle:" << dlerror();
                  }
                }
              }} {
  plugin_handle_ = decltype(plugin_handle_)(
      [&] {
        // Initialize the parameters
        auto const& obj = get<Object>(config);
        std::vector<std::string> kwargs;
        for (auto const& kv : obj) {
          std::string value;
          if (IsA<Integer>(kv.second)) {
            value = std::to_string(get<Integer const>(kv.second));
          } else if (IsA<Number>(kv.second)) {
            value = std::to_string(get<Number const>(kv.second));
          } else if (IsA<Boolean>(kv.second)) {
            value = std::to_string(get<Boolean const>(kv.second));
          } else if (IsA<String>(kv.second)) {
            value = get<String const>(kv.second);
          } else {
            LOG(FATAL) << "Invalid type of federated plugin parameter:"
                       << kv.second.GetValue().TypeStr();
          }
          if (kv.first == "path") {
            continue;
          }
          kwargs.emplace_back(kv.first + "=" + value);
        }
        std::vector<char const*> ckwargs(kwargs.size());
        std::transform(kwargs.cbegin(), kwargs.cend(), ckwargs.begin(),
                       [](auto const& str) { return str.c_str(); });

        // Plugin itself
        PluginCreate_ = SafeLoad<decltype(PluginCreate_)>(plugin_.get(), "FederatedPluginCreate");
        PluginClose_ = SafeLoad<decltype(PluginClose_)>(plugin_.get(), "FederatedPluginClose");
        ErrorMsg_ = SafeLoad<decltype(ErrorMsg_)>(plugin_.get(), "FederatedPluginErrorMsg");

        auto handle =
            this->PluginCreate_(static_cast<std::int32_t>(ckwargs.size()), ckwargs.data());
        CHECK(handle) << "Failed to create federated plugin";

        return handle;
      }(),
      [this](federated::FederatedPluginHandle handle) {
        int rc = this->PluginClose_(handle);
        if (rc != 0) {
          LOG(WARNING) << "Failed to close plugin";
        }
      });
  // gradient
  CHECK(plugin_handle_);
  Encrypt_ = SafeLoad<decltype(Encrypt_)>(plugin_.get(), "FederatedPluginEncryptGPairs");
  SyncEncrypt_ =
      SafeLoad<decltype(SyncEncrypt_)>(plugin_.get(), "FederatedPluginSyncEncryptedGPairs");
  // Vertical
  ResetHistCtxVert_ =
      SafeLoad<decltype(ResetHistCtxVert_)>(plugin_.get(), "FederatedPluginResetHistContextVert");
  BuildEncryptedHistVert_ = SafeLoad<decltype(BuildEncryptedHistVert_)>(
      plugin_.get(), "FederatedPluginBuildEncryptedHistVert");
  SyncEncryptedHistVert_ = SafeLoad<decltype(SyncEncryptedHistVert_)>(
      plugin_.get(), "FederatedPluginSyncEncryptedHistVert");
  // Horizontal
  BuildEncryptedHistHori_ = SafeLoad<decltype(BuildEncryptedHistHori_)>(
      plugin_.get(), "FederatedPluginBuildEncryptedHistHori");
  SyncEncryptedHistHori_ = SafeLoad<decltype(SyncEncryptedHistHori_)>(
      plugin_.get(), "FederatedPluginSyncEncryptedHistHori");
}

FederatedPlugin::~FederatedPlugin() = default;

[[nodiscard]] FederatedPluginBase* CreateFederatedPlugin(Json config) {
  auto plugin = OptionalArg<Object>(config, "federated_plugin", Object::Map{});
  if (!plugin.empty()) {
    auto name_it = plugin.find("name");
    if (name_it != plugin.cend() && get<String const>(name_it->second) == "mock") {
      return new FederatedPluginMock{};
    }
    auto path = get<String>(plugin["path"]);
    return new FederatedPlugin{path, config};
  }
  return nullptr;
}
}  // namespace xgboost::collective
