/*!
 * Copyright 2022 XGBoost contributors
 */
#include <federated.pb.h>

#include "federated_client.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: federated_client server_address(host:port) rank" << '\n';
    return 1;
  }
  auto const server_address = argv[1];
  auto const rank = std::stoi(argv[2]);
  xgboost::federated::FederatedClient client(server_address, rank);

  for (int i = 1; i <= 10; i++) {
    // Allgather.
    std::string allgather_send = "hello " + std::to_string(rank) + ":" + std::to_string(i) + " ";
    auto const allgather_receive = client.Allgather(allgather_send);
    std::cout << "Allgather rank " << rank << ": " << allgather_receive << '\n';

    // Allreduce.
    int data[] = {1 * i, 2 * i, 3 * i, 4 * i, 5 * i};
    int n = sizeof(data) / sizeof(data[0]);
    std::string send_buffer(reinterpret_cast<char const *>(data), sizeof(data));
    auto receive_buffer =
        client.Allreduce(send_buffer, xgboost::federated::INT, xgboost::federated::SUM);
    auto *result = reinterpret_cast<int *>(receive_buffer.data());
    std::cout << "Allreduce rank " << rank << ": ";
    std::copy(result, result + n, std::ostream_iterator<int>(std::cout, " "));
    std::cout << '\n';

    // Broadcast.
    std::string broadcast_send{};
    if (rank == 0) {
      broadcast_send = "hello " + std::to_string(i);
    }
    auto const broadcast_receive = client.Broadcast(broadcast_send, 0);
    std::cout << "Broadcast rank " << rank << ": " << broadcast_receive << '\n';
  }

  return 0;
}
