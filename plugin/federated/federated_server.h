/*!
 * Copyright 2022 XGBoost contributors
 */
namespace xgboost {
namespace federated {

void RunServer(int port, int world_size, char const* server_key_file, char const* server_cert_file,
               char const* client_cert_file);

}  // namespace federated
}  // namespace xgboost
