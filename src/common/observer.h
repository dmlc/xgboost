/*!
 * Copyright 2019 XGBoost contributors
 * \file observer.h
 */
#ifndef XGBOOST_COMMON_OBSERVER_H_
#define XGBOOST_COMMON_OBSERVER_H_

#include <iostream>
#include <string>
#include <vector>

#include "xgboost/host_device_vector.h"
#include "xgboost/parameter.h"
#include "xgboost/json.h"
#include "xgboost/base.h"
#include "xgboost/tree_model.h"

namespace xgboost {
/*\brief  An observer for logging internal data structures.
 *
 *  This class is designed to be `diff` tool friendly, which means it uses plain
 *  `std::cout` for printing to avoid the time information emitted by `LOG(DEBUG)` or
 *  similiar facilities.
 */
class TrainingObserver {
#if defined(XGBOOST_USE_DEBUG_OUTPUT)
  bool constexpr static observe_ {true};
#else
  bool constexpr static observe_ {false};
#endif  // defined(XGBOOST_USE_DEBUG_OUTPUT)

 public:
  void Update(int32_t iter) const {
    if (XGBOOST_EXPECT(!observe_, true)) { return; }
    std::cout << "Iter: " << iter << std::endl;
  }
  /*\brief Observe tree. */
  void Observe(RegTree const& tree) {
    if (XGBOOST_EXPECT(!observe_, true)) { return; }
    std::cout << "Tree:" << std::endl;
    Json j_tree {Object()};
    tree.SaveModel(&j_tree);
    std::string str;
    Json::Dump(j_tree, &str, true);
    std::cout << str << std::endl;
  }
  /*\brief Observe tree. */
  void Observe(RegTree const* p_tree) {
    if (XGBOOST_EXPECT(!observe_, true)) { return; }
    auto const& tree = *p_tree;
    this->Observe(tree);
  }
  /*\brief Observe data hosted by `std::vector'. */
  template <typename T>
  void Observe(std::vector<T> const& h_vec, std::string name) const {
    if (XGBOOST_EXPECT(!observe_, true)) { return; }
    std::cout << "Procedure: " << name << std::endl;

    for (size_t i = 0; i < h_vec.size(); ++i) {
      std::cout << h_vec[i] << ", ";
      if (i % 8 == 0) {
        std::cout << '\n';
      }
    }
    std::cout << std::endl;
  }
  /*\brief Observe data hosted by `HostDeviceVector'. */
  template <typename T>
  void Observe(HostDeviceVector<T> const& vec, std::string name) const {
    if (XGBOOST_EXPECT(!observe_, true)) { return; }
    auto const& h_vec = vec.HostVector();
    this->Observe(h_vec, name);
  }
  /*\brief Observe objects with `XGBoostParamer' type. */
  template <typename Parameter,
            typename std::enable_if<
              std::is_base_of<XGBoostParameter<Parameter>, Parameter>::value>::type* = nullptr>
  void Observe(const Parameter &p, std::string name) const {
    if (XGBOOST_EXPECT(!observe_, true)) { return; }

    Json obj {toJson(p)};
    std::cout << "Parameter: " << name << ":\n" << obj << std::endl;
  }
  /*\brief Observe parameters provided by users. */
  void Observe(Args const& args) const {
    if (XGBOOST_EXPECT(!observe_, true)) { return; }

    for (auto kv : args) {
      std::cout << kv.first << ": " << kv.second << "\n";
    }
    std::cout << std::endl;
  }

  /*\brief Get a global instance. */
  static TrainingObserver& Instance() {
    static TrainingObserver observer;
    return observer;
  }
};
}  // namespace xgboost
#endif  // XGBOOST_COMMON_OBSERVER_H_
