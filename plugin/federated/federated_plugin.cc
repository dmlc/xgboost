/**
 * Copyright 2024, XGBoost Contributors
 */
#include "federated_plugin.h"

#include <dlfcn.h>  // for dlclose, dlsym, dlopen

#include <cstddef>  // for size_t
#include <cstdint>  // for uint8_t, uint64_t
#include <sstream>  // for stringstream

#include "xgboost/base.h"         // for bst_bin_t, bst_feature_t
#include "xgboost/json.h"         // for Json
#include "xgboost/linalg.h"       // for MakeTensorView
#include "xgboost/logging.h"      // for CHECK_EQ
#include "xgboost/span.h"         // for Span
#include "xgboost/string_view.h"  // for StringView
#include "../../src/common/json_utils.h"  // for OptionalArg

namespace xgboost::collective {
void FederatedPluginMock::Reset(common::Span<std::uint32_t const> cutptrs,
                                common::Span<std::int32_t const> bin_idx) {
  this->cuts_.resize(cutptrs.size());
  std::copy_n(cutptrs.data(), cutptrs.size(), this->cuts_.data());
  this->gidx_.resize(bin_idx.size());
  std::copy_n(bin_idx.data(), bin_idx.size(), this->gidx_.data());
}

[[nodiscard]] common::Span<std::uint8_t> FederatedPluginMock::BuildEncryptedHistVert(
    common::Span<std::uint64_t const*> rowptrs, common::Span<std::size_t const> sizes,
    common::Span<bst_node_t const> nids) {
  bst_bin_t total_bin_size = cuts_.back();
  bst_feature_t n_features = cuts_.size() - 1;
  bst_idx_t n_total_samples = gidx_.size() / n_features;
  CHECK_EQ(gidx_.size() % n_features, 0);
  bst_bin_t hist_size = total_bin_size * 2;
  hist_plain_.resize(hist_size);
  CHECK_EQ(rowptrs.size(), sizes.size());
  CHECK_EQ(nids.size(), sizes.size());
  CHECK_EQ(n_total_samples * 2, grad_.size());

  Context ctx;
  auto gidx = linalg::MakeTensorView(&ctx, common::Span{gidx_.data(), gidx_.size()},
                                     n_total_samples, n_features);
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    auto n_samples = sizes[i];
    auto samples = common::Span{rowptrs[i], n_samples};
    for (auto ridx : samples) {
      for (bst_feature_t f = 0; f < n_features; ++f) {
        auto bin = gidx(ridx, f);
        if (bin < 0) {
          continue;
        }
        auto g = grad_[ridx * 2];
        auto h = grad_[ridx * 2 + 1];
        hist_plain_[bin * 2] += g;
        hist_plain_[bin * 2 + 1] += h;
      }
    }
  }

  return {reinterpret_cast<std::uint8_t*>(hist_plain_.data()),
          common::Span<double>{hist_plain_}.size_bytes()};
}

[[nodiscard]] common::Span<double> FederatedPluginMock::SyncEncryptedHistVert(
    common::Span<std::uint8_t>) {
  return {hist_plain_};
}

template <typename T>  // fixme, duplicated code
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
      plugin_.get(), "FederatedPluginSyncEnrcyptedHistVert");
  // Horizontal
  BuildEncryptedHistHori_ = SafeLoad<decltype(BuildEncryptedHistHori_)>(
      plugin_.get(), "FederatedPluginBuildEncryptedHistHori");
  SyncEncryptedHistHori_ = SafeLoad<decltype(SyncEncryptedHistHori_)>(
      plugin_.get(), "FederatedPluginSyncEnrcyptedHistHori");
}

FederatedPlugin::~FederatedPlugin() = default;

[[nodiscard]] FederatedPluginBase* CreateFederatedPlugin(Json config) {
  auto plugin = OptionalArg<Object>(config, "federated_plugin", Object::Map{});
  if (!plugin.empty()) {
    auto path = get<String>(plugin["path"]);
    return new FederatedPlugin{path, config};
  }
  return new FederatedPluginMock{};
}
}  // namespace xgboost::collective
