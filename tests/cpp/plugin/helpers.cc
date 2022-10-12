#include <chrono>
#include <thread>
#include <random>
#include <cstdint>

#include "helpers.h"

using namespace std::chrono_literals;

int GenerateRandomPort(int low, int high) {
  // Ensure unique timestamp by introducing a small artificial delay
  std::this_thread::sleep_for(100ms);
  auto timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count());
  std::mt19937_64 rng(timestamp);
  std::uniform_int_distribution<int> dist(low, high);
  int port = dist(rng);
  return port;
}
