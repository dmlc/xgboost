/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_C_API_C_API_UTILS_H_
#define XGBOOST_C_API_C_API_UTILS_H_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>  // std::shared_ptr
#include <string>
#include <vector>

#include "xgboost/c_api.h"
#include "xgboost/data.h"         // DMatrix
#include "xgboost/feature_map.h"  // for FeatureMap
#include "xgboost/json.h"
#include "xgboost/learner.h"
#include "xgboost/linalg.h"       // ArrayInterfaceHandler
#include "xgboost/logging.h"
#include "xgboost/string_view.h"  // StringView

namespace xgboost {
/* \brief Determine the output shape of prediction.
 *
 * \param strict_shape Whether should we reshape the output with consideration of groups
 *                     and forest.
 * \param type         Prediction type
 * \param rows         Input samples
 * \param cols         Input features
 * \param chunksize    Total elements of output / rows
 * \param groups       Number of output groups from Learner
 * \param rounds       end_iteration - beg_iteration
 * \param out_shape    Output shape
 * \param out_dim      Output dimension
 */
inline void CalcPredictShape(bool strict_shape, PredictionType type, size_t rows, size_t cols,
                             size_t chunksize, size_t groups, size_t rounds,
                             std::vector<bst_ulong> *out_shape,
                             xgboost::bst_ulong *out_dim) {
  auto &shape = *out_shape;
  if (type == PredictionType::kMargin && rows != 0) {
    // When kValue is used, softmax can change the chunksize.
    CHECK_EQ(chunksize, groups);
  }

  switch (type) {
  case PredictionType::kValue:
  case PredictionType::kMargin: {
    if (chunksize == 1 && !strict_shape) {
      *out_dim = 1;
      shape.resize(*out_dim);
      shape.front() = rows;
    } else {
      *out_dim = 2;
      shape.resize(*out_dim);
      shape.front() = rows;
      // chunksize can be 1 if it's softmax
      shape.back() = std::min(groups, chunksize);
    }
    break;
  }
  case PredictionType::kApproxContribution:
  case PredictionType::kContribution: {
    if (groups == 1 && !strict_shape) {
      *out_dim = 2;
      shape.resize(*out_dim);
      shape.front() = rows;
      shape.back() = cols + 1;
    } else {
      *out_dim = 3;
      shape.resize(*out_dim);
      shape[0] = rows;
      shape[1] = groups;
      shape[2] = cols + 1;
    }
    break;
  }
  case PredictionType::kApproxInteraction:
  case PredictionType::kInteraction: {
    if (groups == 1 && !strict_shape) {
      *out_dim = 3;
      shape.resize(*out_dim);
      shape[0] = rows;
      shape[1] = cols + 1;
      shape[2] = cols + 1;
    } else {
      *out_dim = 4;
      shape.resize(*out_dim);
      shape[0] = rows;
      shape[1] = groups;
      shape[2] = cols + 1;
      shape[3] = cols + 1;
    }
    break;
  }
  case PredictionType::kLeaf: {
    if (strict_shape) {
      shape.resize(4);
      shape[0] = rows;
      shape[1] = rounds;
      shape[2] = groups;
      auto forest = chunksize / (shape[1] * shape[2]);
      forest = std::max(static_cast<decltype(forest)>(1), forest);
      shape[3] = forest;
      *out_dim = shape.size();
    } else if (chunksize == 1) {
      *out_dim = 1;
      shape.resize(*out_dim);
      shape.front() = rows;
    } else {
      *out_dim = 2;
      shape.resize(*out_dim);
      shape.front() = rows;
      shape.back() = chunksize;
    }
    break;
  }
  default: {
    LOG(FATAL) << "Unknown prediction type:" << static_cast<int>(type);
  }
  }
  CHECK_EQ(
      std::accumulate(shape.cbegin(), shape.cend(), static_cast<bst_ulong>(1), std::multiplies<>{}),
      chunksize * rows);
}

// Reverse the ntree_limit in old prediction API.
inline uint32_t GetIterationFromTreeLimit(uint32_t ntree_limit, Learner *learner) {
  // On Python and R, `best_ntree_limit` is set to `best_iteration * num_parallel_tree`.
  // To reverse it we just divide it by `num_parallel_tree`.
  if (ntree_limit != 0) {
    learner->Configure();
    uint32_t num_parallel_tree = 0;

    Json config{Object()};
    learner->SaveConfig(&config);
    auto const &booster = get<String const>(config["learner"]["gradient_booster"]["name"]);
    if (booster == "gblinear") {
      num_parallel_tree = 0;
    } else if (booster == "dart") {
      num_parallel_tree =
          std::stoi(get<String const>(config["learner"]["gradient_booster"]["gbtree"]
                                            ["gbtree_model_param"]["num_parallel_tree"]));
    } else if (booster == "gbtree") {
      num_parallel_tree = std::stoi(get<String const>(
          (config["learner"]["gradient_booster"]["gbtree_model_param"]["num_parallel_tree"])));
    } else {
      LOG(FATAL) << "Unknown booster:" << booster;
    }
    ntree_limit /= std::max(num_parallel_tree, 1u);
  }
  return ntree_limit;
}

inline float GetMissing(Json const &config) {
  float missing;
  auto const &obj = get<Object const>(config);
  auto it = obj.find("missing");
  if (it == obj.cend()) {
    LOG(FATAL) << "Argument `missing` is required.";
  }

  auto const &j_missing = it->second;
  if (IsA<Number const>(j_missing)) {
    missing = get<Number const>(j_missing);
  } else if (IsA<Integer const>(j_missing)) {
    missing = get<Integer const>(j_missing);
  } else {
    missing = nan("");
    TypeCheck<Number, Integer>(j_missing, "missing");
  }
  return missing;
}

// Safe guard some global variables from being changed by XGBoost.
class XGBoostAPIGuard {
#if defined(XGBOOST_USE_CUDA)
  int32_t device_id_ {0};

  void SetGPUAttribute();
  void RestoreGPUAttribute();
#else
  void SetGPUAttribute() {}
  void RestoreGPUAttribute() {}
#endif

 public:
  XGBoostAPIGuard() {
    SetGPUAttribute();
  }
  ~XGBoostAPIGuard() {
    RestoreGPUAttribute();
  }
};

inline FeatureMap LoadFeatureMap(std::string const& uri) {
  FeatureMap feat;
  if (uri.size() != 0) {
    std::unique_ptr<dmlc::Stream> fs(dmlc::Stream::Create(uri.c_str(), "r"));
    dmlc::istream is(fs.get());
    feat.LoadText(is);
  }
  return feat;
}

inline void GenerateFeatureMap(Learner const *learner,
                               std::vector<Json> const &custom_feature_names,
                               size_t n_features, FeatureMap *out_feature_map) {
  auto &feature_map = *out_feature_map;
  auto maybe = [&](std::vector<std::string> const &values, size_t i,
                   std::string const &dft) {
    return values.empty() ? dft : values[i];
  };
  if (feature_map.Size() == 0) {
    // Use the feature names and types from booster.
    std::vector<std::string> feature_names;
    // priority:
    // 1. feature map.
    // 2. customized feature name.
    // 3. from booster
    // 4. default feature name.
    if (!custom_feature_names.empty()) {
      CHECK_EQ(custom_feature_names.size(), n_features)
          << "Incorrect number of feature names.";
      feature_names.resize(custom_feature_names.size());
      std::transform(custom_feature_names.begin(), custom_feature_names.end(),
                     feature_names.begin(),
                     [](Json const &name) { return get<String const>(name); });
    } else {
      learner->GetFeatureNames(&feature_names);
    }
    if (!feature_names.empty()) {
      CHECK_EQ(feature_names.size(), n_features) << "Incorrect number of feature names.";
    }

    std::vector<std::string> feature_types;
    learner->GetFeatureTypes(&feature_types);
    if (!feature_types.empty()) {
      CHECK_EQ(feature_types.size(), n_features) << "Incorrect number of feature types.";
    }

    for (size_t i = 0; i < n_features; ++i) {
      feature_map.PushBack(
          i,
          maybe(feature_names, i, "f" + std::to_string(i)).data(),
          maybe(feature_types, i, "q").data());
    }
  }
  CHECK_EQ(feature_map.Size(), n_features);
}

void XGBBuildInfoDevice(Json* p_info);

template <typename JT>
auto const &RequiredArg(Json const &in, StringView key, StringView func) {
  auto const &obj = get<Object const>(in);
  auto it = obj.find(key);
  if (it == obj.cend() || IsA<Null>(it->second)) {
    LOG(FATAL) << "Argument `" << key << "` is required for `" << func << "`.";
  }
  TypeCheck<JT>(it->second, StringView{key});
  return get<std::remove_const_t<JT> const>(it->second);
}

template <typename JT, typename T>
auto const &OptionalArg(Json const &in, StringView key, T const &dft) {
  auto const &obj = get<Object const>(in);
  auto it = obj.find(key);
  if (it != obj.cend() && !IsA<Null>(it->second)) {
    TypeCheck<JT>(it->second, key);
    return get<std::remove_const_t<JT> const>(it->second);
  }
  return dft;
}

/**
 * \brief Get shared ptr from DMatrix C handle with additional checks.
 */
inline std::shared_ptr<DMatrix> CastDMatrixHandle(DMatrixHandle const handle) {
  auto pp_m = static_cast<std::shared_ptr<DMatrix> *>(handle);
  StringView msg{"Invalid DMatrix handle"};
  CHECK(pp_m) << msg;
  auto p_m = *pp_m;
  CHECK(p_m) << msg;
  return p_m;
}

namespace detail {
template <typename PtrT, typename I, typename T>
void MakeSparseFromPtr(PtrT const *p_indptr, I const *p_indices, T const *p_data,
                       std::size_t nindptr, std::string *indptr_str, std::string *indices_str,
                       std::string *data_str) {
  auto ndata = static_cast<Integer::Int>(p_indptr[nindptr - 1]);
  // Construct array interfaces
  Json jindptr{Object{}};
  Json jindices{Object{}};
  Json jdata{Object{}};
  CHECK(p_indptr);
  jindptr["data"] =
      Array{std::vector<Json>{Json{reinterpret_cast<Integer::Int>(p_indptr)}, Json{true}}};
  jindptr["shape"] = std::vector<Json>{Json{nindptr}};
  jindptr["version"] = Integer{3};

  CHECK(p_indices);
  jindices["data"] =
      Array{std::vector<Json>{Json{reinterpret_cast<Integer::Int>(p_indices)}, Json{true}}};
  jindices["shape"] = std::vector<Json>{Json{ndata}};
  jindices["version"] = Integer{3};

  CHECK(p_data);
  jdata["data"] =
      Array{std::vector<Json>{Json{reinterpret_cast<Integer::Int>(p_data)}, Json{true}}};
  jdata["shape"] = std::vector<Json>{Json{ndata}};
  jdata["version"] = Integer{3};

  std::string pindptr_typestr =
      linalg::detail::ArrayInterfaceHandler::TypeChar<PtrT>() + std::to_string(sizeof(PtrT));
  std::string ind_typestr =
      linalg::detail::ArrayInterfaceHandler::TypeChar<I>() + std::to_string(sizeof(I));
  std::string data_typestr =
      linalg::detail::ArrayInterfaceHandler::TypeChar<T>() + std::to_string(sizeof(T));
  if (DMLC_LITTLE_ENDIAN) {
    jindptr["typestr"] = String{"<" + pindptr_typestr};
    jindices["typestr"] = String{"<" + ind_typestr};
    jdata["typestr"] = String{"<" + data_typestr};
  } else {
    jindptr["typestr"] = String{">" + pindptr_typestr};
    jindices["typestr"] = String{">" + ind_typestr};
    jdata["typestr"] = String{">" + data_typestr};
  }

  Json::Dump(jindptr, indptr_str);
  Json::Dump(jindices, indices_str);
  Json::Dump(jdata, data_str);
}
}  // namespace detail
}  // namespace xgboost
#endif  // XGBOOST_C_API_C_API_UTILS_H_
