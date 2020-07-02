/*!
 * Copyright 2020 by XGBoost Contributors
 * \file updater_multi_exact.h
 * \brief Implementation of exact tree method for training multi-target trees.
 */
#ifndef XGBOOST_TREE_UPDATER_EXACT_H_
#define XGBOOST_TREE_UPDATER_EXACT_H_

#include <limits>
#include <vector>
#include <functional>

#include "xgboost/tree_updater.h"
#include "xgboost/tree_model.h"
#include "xgboost/json.h"
#include "param.h"
#include "constraints.h"
#include "../common/random.h"
#include "../common/timer.h"

namespace xgboost {
/* \brief A simple wrapper around `std::vector`.  Not much numeric computation is
 * needed for XGBoost so we just hand fusing all vector operations without using
 * expression template.
 */
template <typename Type>
struct Vector {
  using ValueT = Type;

  std::vector<Type> vec;

  Vector() = default;
  explicit Vector(size_t n, Type v = Type{}) : vec(n, v) {}  // NOLINT

  void Resize(size_t s) { return vec.resize(s); }

  size_t Size() const { return vec.size(); }

  template <typename Op>
  Vector BinaryOp(Vector const& that, Op op) const {
    CHECK_EQ(vec.size(), that.vec.size());
    Vector ret(that.vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
      ret.vec[i] = op(vec[i], that.vec[i]);
    }
    return ret;
  }
  template <typename Op>
  Vector BinaryScalar(Type const& that, Op op) const {
    Vector ret(Size());
    for (size_t i = 0; i < vec.size(); ++i) {
      ret[i] = op(vec[i], that);
    }
    return ret;
  }

  Vector operator+(Vector const& that) const {
    return BinaryOp(that, std::plus<Type>());
  }
  Vector operator+(Type const& that) const {
    return BinaryScalar(that, std::plus<Type>());
  }
  Vector& operator+=(Vector const& that) {
    size_t size = vec.size();
    for (size_t i = 0; i < size; ++i) {
      vec[i] += that[i];
    }
    return *this;
  }
  Vector operator*(Type const& that) const {
    return BinaryScalar(that, std::multiplies<Type>());
  }

  bool operator==(Vector const& that) const {
    if (Size() != that.Size()) {
      return false;
    }
    for (size_t i = 0; i < that.Size(); ++i) {
      if (vec[i] != that.vec[i]) {
        return false;
      }
    }
    return true;
  }

  Type const& operator[](size_t i) const {
    return vec[i];
  }
  Type& operator[](size_t i) {
    return vec[i];
  }

  friend std::ostream& operator<<(std::ostream& os, Vector vec)  {
    for (size_t i = 0; i < vec.Size(); ++i) {
      os << vec[i];
      if (i != vec.Size() - 1) {
        os << ", ";
      }
    }
    return os;
  }
};

/*\brief A specialization over Vector for scalar value.  This can be twice
 * faster. */
template <typename Type>
struct ScalarContainer {
  using ValueT = Type;
  static_assert(std::is_floating_point<ValueT>::value, "");
  ValueT vec;

 public:
  constexpr ScalarContainer() = default;
  constexpr explicit ScalarContainer(size_t, ValueT v) : vec{v} {}
  void Resize(size_t) const { }
  constexpr ScalarContainer(ValueT v) : vec{v} {} // NOLINT
  ScalarContainer& operator+=(ValueT v) {
    vec += v;
    return *this;
  }
  ScalarContainer& operator-=(ValueT v) {
    vec -= v;
    return *this;
  }
  constexpr ScalarContainer operator+(ScalarContainer v) const {
    return ScalarContainer{vec + v.vec};
  }
  constexpr ScalarContainer operator*(ScalarContainer v) const {
    return ScalarContainer{vec * v.vec};
  }

  ScalarContainer &operator+=(ScalarContainer v) {
    vec += v.vec;
    return *this;
  }
  constexpr ValueT const& operator[](size_t) const { return vec; }
  ValueT& operator[](size_t) { return vec; }

  constexpr size_t Size() const { return 1; }

  friend std::ostream& operator<<(std::ostream& os, ScalarContainer const& s) {
    os << s.vec;
    return os;
  }
};

using Scalar = ScalarContainer<double>;

static_assert(std::is_pod<Scalar>::value, "");
static_assert(std::alignment_of<Scalar>::value ==
                  std::alignment_of<double>::value,
              "");
static_assert(sizeof(Scalar) == sizeof(double), "");

using MultiGradientPair = detail::GradientPairInternal<Vector<double>>;
using SingleGradientPair = detail::GradientPairInternal<Scalar>;

template <typename T>
bool AnyLT(T const& vec, float v) {
  // A very time consuming loop.
  for (size_t i = 0; i < vec.Size(); ++i) {
    if (vec[i] < v) {
      return true;
    }
  }
  return false;
}

template <>
inline bool AnyLT<Scalar>(Scalar const& value, float v) {
  return value.vec < v;
}

template <typename T>
bool AnyLE(T const& lhs, T const& rhs) {
  CHECK_EQ(lhs.Size(), rhs.Size());
  for (size_t i = 0; i < lhs.Size(); ++i) {
    if (lhs[i] <= rhs[i]) {
      return true;
    }
  }
  return false;
}

namespace tree {
template <typename GradientT> struct WeightType {
  using Type = typename std::conditional<
      std::is_same<GradientT, MultiGradientPair>::value, Vector<float>,
      typename std::conditional<
          std::is_same<GradientT, MultiGradientPair::ValueT>::value,
          Vector<float>, ScalarContainer<float>>::type
      >::type;
};
template <typename GradientT>
using WeightT = typename WeightType<GradientT>::Type;

template <typename GradientT, typename ReturnT = WeightT<GradientT>>
ReturnT MultiCalcWeight(GradientT const &sum_grad, GradientT const &sum_hess,
                        TrainParam const &p) {
  ReturnT w(sum_grad.Size(), 0);
  for (size_t i = 0; i < w.Size(); ++i) {
    w[i] = CalcWeight(p, sum_grad[i], sum_hess[i]);
  }
  return w;
}

template <>
inline ScalarContainer<float> MultiCalcWeight<Scalar, ScalarContainer<float>>(
    Scalar const &sum_grad, Scalar const &sum_hess, TrainParam const &p) {
  return CalcWeight(p, sum_grad.vec, sum_hess.vec);
}

template <typename GradientT, typename ReturnT = WeightT<GradientT>>
ReturnT MultiCalcWeight(GradientT const &sum_gradients, TrainParam const &p) {
  return MultiCalcWeight(sum_gradients.GetGrad(), sum_gradients.GetHess(), p);
}

template <typename GradientT, typename Weight>
float MultiCalcGainGivenWeight(GradientT const &sum_grad,
                               GradientT const &sum_hess,
                               Weight const& weight,
                               TrainParam const &p) {
  float gain { 0 };
  for (size_t i = 0; i < weight.Size(); ++i) {
    gain += -weight[i] * ThresholdL1(sum_grad[i], p.reg_alpha);
  }
  return gain;
}

template <>
inline float MultiCalcGainGivenWeight<Scalar, ScalarContainer<float>>(
    Scalar const &sum_grad, Scalar const &sum_hess,
    ScalarContainer<float> const &weight, TrainParam const &p) {
  return CalcGainGivenWeight(p, sum_grad.vec, sum_hess.vec,
                             static_cast<double>(weight.vec));
}

template <typename GradientT>
inline GradientT MakeGradientPair(size_t columns);
template <>
inline SingleGradientPair MakeGradientPair<SingleGradientPair>(size_t) {
  return SingleGradientPair{};
}
template <>
inline MultiGradientPair MakeGradientPair<MultiGradientPair>(size_t columns) {
  return MultiGradientPair(Vector<MultiGradientPair::ValueT::ValueT>(columns, 0),
                           Vector<MultiGradientPair::ValueT::ValueT>(columns, 0));
}

template <typename GradientT>
struct ExactSplitEntryContainer {
  using Candidate = SplitEntryContainer<GradientT>;
  Candidate candidate;
  float last_value = {std::numeric_limits<float>::quiet_NaN()};

  bst_node_t nidx { 0 };
  GradientT parent_sum;
  float root_gain { 0 };

  ExactSplitEntryContainer() = default;
  explicit ExactSplitEntryContainer(bst_node_t nidx,
                                    GradientT const &gradient_sum,
                                    float root_gain,
                                    TrainParam const &p)
      : nidx{nidx}, parent_sum{gradient_sum}, root_gain{root_gain} {
  }

  bool IsValid(int32_t depth, int32_t leaves, TrainParam const& p) {
    if (candidate.loss_chg <= kRtEps) { return false; }
    if (AnyLT(candidate.left_sum.GetHess(), p.min_child_weight)) {
      return false;
    }
    if (candidate.loss_chg < p.min_split_loss) {
      return false;
    }
    if (p.max_depth > 0 && depth >= p.max_depth) { return false; }
    if (p.max_leaves > 0 && leaves >= p.max_leaves) { return false; }
    return true;
  }

  void Accumulate(GradientT const& g, float value) {
    this->candidate.left_sum += g;
    this->last_value = value;
  }

  static bool ChildIsValid(int32_t depth, int32_t leaves, TrainParam const& p) {
    if (p.max_depth > 0 && depth >= p.max_depth) {
      return false;
    }
    if (p.max_leaves > 0 && leaves >= p.max_leaves) {
      return false;
    }
    return true;
  }
};

template <typename Type>
class MultiValueConstraint {
  common::Span<float> lower_;
  common::Span<float> upper_;
  common::Span<int32_t const> monotone_;
  size_t targets_;

 public:
  struct Storage {
    HostDeviceVector<float> lower_storage;
    HostDeviceVector<float> upper_storage;
  };

  void Init(TrainParam const& p, size_t targets, Storage* storage) {
    targets_ = targets;
    monotone_ = {p.monotone_constraints};
    if (!monotone_.empty()) {
      storage->lower_storage.Resize(p.MaxNodes(),
                                    -std::numeric_limits<float>::max());
      storage->upper_storage.Resize(p.MaxNodes(),
                                    std::numeric_limits<float>::max());
      lower_ = storage->lower_storage.HostSpan();
      upper_ = storage->upper_storage.HostSpan();
    }
  }

  template <typename GradientT>
  Type CalcWeight(GradientT const &sum_gradients, bst_node_t nidx,
                  TrainParam const &p) const {
    auto weight =
        MultiCalcWeight(sum_gradients.GetGrad(), sum_gradients.GetHess(), p);
    static_assert(std::is_same<typename decltype(weight)::ValueT, float>::value,
                  "");
    if (monotone_.empty()) {
      return weight;
    }
    CHECK_GT(targets_, 0);
    auto lower = lower_.subspan(nidx * targets_, targets_);
    auto upper = upper_.subspan(nidx * targets_, targets_);
    for (size_t i = 0; i < weight.Size(); ++i) {
      auto &w = weight[i];
      if (w < lower[i]) {
        w = lower[i];
      } else if (w > upper[i]) {
        w = upper[i];
      }
    }
    return weight;
  }

  template <typename GradientT>
  float CalcSplitGain(GradientT const &left, GradientT const &right,
                      bst_node_t nidx, bst_feature_t feature_id,
                      TrainParam const &p) const {
    auto left_weight = this->CalcWeight(left, nidx, p);
    auto right_weight = this->CalcWeight(right, nidx, p);
    typename Type::ValueT gain =
        MultiCalcGainGivenWeight(left.GetGrad(), left.GetHess(), left_weight,
                                 p) +
        MultiCalcGainGivenWeight(right.GetGrad(), right.GetHess(), right_weight,
                                 p);
    if (monotone_.empty()) {
      return gain;
    }

    int32_t constraint =
        feature_id >= monotone_.size() ? 0 : monotone_[feature_id];
    if (constraint > 0) {
      if (!AnyLE(left_weight, right_weight)) {
        gain = -std::numeric_limits<float>::infinity();
      }
    } else if (constraint < 0) {
      if (!AnyLE(right_weight, left_weight)) {
        gain = -std::numeric_limits<float>::infinity();
      }
    }
    return gain;
  }

  void Split(bst_node_t nidx,
             bst_node_t left, Type const &left_weight,
             bst_node_t right, Type const &right_weight,
             bst_feature_t feature_id) {
    if (monotone_.empty()) {
      return;
    }

    Type mid {left_weight.Size(), 0.0};
    for (size_t i = 0; i < mid.Size(); ++i) {
      mid[i] = (left_weight[i] + right_weight[i]) / 2;
      CHECK(!std::isnan(mid[i]));
    }
    int32_t constraint = monotone_[feature_id];

    auto upper_left = upper_.subspan(left * targets_, targets_);
    auto lower_left = lower_.subspan(left * targets_, targets_);

    auto upper_right = upper_.subspan(right * targets_, targets_);
    auto lower_right = lower_.subspan(right * targets_, targets_);

    auto upper_parent = upper_.subspan(nidx * targets_, targets_);
    auto lower_parent = lower_.subspan(nidx * targets_, targets_);

    for (size_t i = 0; i < mid.Size(); ++i) {
      upper_left[i] = upper_parent[i];
      upper_right[i] = upper_parent[i];

      lower_left[i] = lower_parent[i];
      lower_right[i] = lower_parent[i];
    }

    if (constraint < 0) {
      for (size_t i = 0; i < mid.Size(); ++i) {
        lower_left[i] = mid[i];
        upper_right[i] = mid[i];
      }
    } else if (constraint > 0) {
      for (size_t i = 0; i < mid.Size(); ++i) {
        upper_left[i] = mid[i];
        lower_right[i] = mid[i];
      }
    }
  }
};

template <typename GradientT>
class MultiExact : public TreeUpdater {
 protected:
  using SplitEntry = ExactSplitEntryContainer<GradientT>;
  // A copy of gradients.
  std::vector<GradientT> gpairs_;
  // Maps row idx to tree node idx.
  std::vector<bst_node_t> positions_;
  // When a node can be further splited. 0 means no, 1 means yes.
  // std::vector<bool> is not thread safe.
  std::vector<char> is_splitable_;
  // Splits for current tree.
  std::vector<SplitEntry> nodes_split_;
  // Pointer to current layer of nodes.
  size_t node_shift_;
  // Scan of gradient statistic, used in enumeration.
  std::vector<std::vector<SplitEntry>> tloc_scans_;
  common::ColumnSampler sampler_;
  size_t targets_{0};

  bool NeedForward(SparsePage::Inst const &column, TrainParam const& p) const {
    return p.default_direction == 2 ||
           ((column.size() != gpairs_.size()) &&  // with missing
            !(column.size() != 0 &&
              column[0].fvalue == column[column.size() - 1].fvalue));
  }
  bool NeedBackward(SparsePage::Inst const &column, TrainParam const& p) const {
    return p.default_direction != 2;
  }

 public:
  explicit MultiExact() {
    node_shift_ = 0;
    monitor_.Init(__func__);
  }
  void Configure(const Args& args) override {
    param_.UpdateAllowUnknown(args);
    if (param_.grow_policy != TrainParam::kDepthWise) {
      LOG(WARNING) << "Exact tree method supports only depth wise grow policy.";
    }
  }
  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
  }

  void InitData(DMatrix* data, common::Span<GradientPair const> gpairs, size_t targets);
  void InitRoot(DMatrix* data, RegTree* tree);

  void EvaluateFeature(bst_feature_t fid, SparsePage::Inst const &column,
                       std::vector<SplitEntry>* p_scans,
                       std::vector<SplitEntry> *p_nodes) const;
  void EvaluateSplit(DMatrix *data, common::Span<bst_feature_t const> features);

  size_t ExpandTree(RegTree* p_tree, std::vector<SplitEntry>* next);
  void ApplySplit(DMatrix* m, RegTree* p_tree);

  void UpdateTree(HostDeviceVector<GradientPair>* gpair,
                  DMatrix* data, RegTree* tree);

 public:
  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* data,
              const std::vector<RegTree*>& trees) override {
    interaction_constraints_.Configure(param_, data->Info().num_row_);
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    for (auto p_tree : trees) {
      this->UpdateTree(gpair, data, p_tree);
    }
    param_.learning_rate = lr;
  }
  char const* Name() const override { return "grow_colmaker"; };

 private:
  common::Monitor monitor_;
  FeatureInteractionConstraintHost interaction_constraints_;
  MultiValueConstraint<WeightT<GradientT>> value_constraints_;
  typename MultiValueConstraint<WeightT<GradientT>>::Storage
      monotone_constriants_;
  TrainParam param_;
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_EXACT_H_
