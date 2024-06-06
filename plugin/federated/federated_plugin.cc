/**
 * Copyright 2024, XGBoost Contributors
 */
#include "federated_plugin.h"

#include <dlfcn.h>  // for dlclose, dlsym, dlopen

#include "xgboost/json.h"  // for Json
#include "xgboost/logging.h"

typedef void* FederatedPluginHandle;  // NOLINT

namespace xgboost::collective {
template <typename T>  // fixme, duplicated code
auto SafeLoad(FederatedPluginHandle handle, StringView name) {
  std::stringstream errs;
  auto ptr = reinterpret_cast<T>(dlsym(handle, name.c_str()));
  if (!ptr) {
    errs << "Failed to load symbol `" << name << "`. Error:\n  " << dlerror();
    LOG(FATAL) << errs.str();
  }
  return ptr;
}

FederatedPlugin::FederatedPlugin(std::string_view path, Json config)
    : plugin_{[&] {
                auto handle = dlopen(path.data(), RTLD_LAZY | RTLD_GLOBAL);
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
      [this](FederatedPluginHandle handle) {
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
}  // namespace xgboost::collective
