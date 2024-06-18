/**
 * Copyright 2024, XGBoost contributors
 */
#include "communicator-inl.h"

namespace xgboost::collective {
[[nodiscard]] std::vector<std::vector<char>> VectorAllgatherV(
    std::vector<std::vector<char>> const &input) {
  auto n_inputs = input.size();
  std::vector<std::int64_t> sizes(n_inputs);
  std::transform(input.cbegin(), input.cend(), sizes.begin(),
                 [](auto const &vec) { return vec.size(); });

  std::vector<std::int64_t> global_sizes = AllgatherV(sizes);
  std::vector<std::int64_t> offset(global_sizes.size() + 1);
  offset[0] = 0;
  for (std::size_t i = 1; i < offset.size(); i++) {
    offset[i] = offset[i - 1] + global_sizes[i - 1];
  }

  std::vector<char> collected;
  for (auto const &vec : input) {
    collected.insert(collected.end(), vec.cbegin(), vec.cend());
  }
  auto out = AllgatherV(collected);

  std::vector<std::vector<char>> result;
  for (std::size_t i = 1; i < offset.size(); ++i) {
    std::vector<char> local(out.cbegin() + offset[i - 1], out.cbegin() + offset[i]);
    result.emplace_back(std::move(local));
  }
  return result;
}
}  // namespace xgboost::collective
