/**
 * Copyright 2019-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <set>
#include <string>

#include "../../../src/common/device_helpers.cuh"
#include "../../../src/tree/constraints.cuh"
#include "../../../src/tree/constraints.h"
#include "../../../src/tree/param.h"
#include "../helpers.h"  // MakeCUDACtx

namespace xgboost {
namespace {

struct FConstraintWrapper : public FeatureInteractionConstraintDevice {
  common::Span<LBitField64> GetNodeConstraints() {
    return FeatureInteractionConstraintDevice::s_node_constraints_;
  }
  FConstraintWrapper(tree::TrainParam param, bst_feature_t n_features)
      : FeatureInteractionConstraintDevice(param, n_features) {}

  dh::device_vector<bst_feature_t> const& GetDSets() const { return d_sets_; }
  std::vector<size_t> const& GetSetsPtr() const { return h_sets_ptr_; }
};

std::string GetConstraintsStr() {
  std::string const constraints_str = R"constraint([[1, 2], [3, 4, 5]])constraint";
  return constraints_str;
}

tree::TrainParam GetParameter() {
  Args args{{"interaction_constraints", GetConstraintsStr()}};
  tree::TrainParam param;
  param.Init(args);
  return param;
}

void CompareBitField(LBitField64 d_field, std::set<uint32_t> positions) {
  std::vector<LBitField64::value_type> h_field_storage(d_field.Bits().size());
  thrust::copy(
      thrust::device_ptr<LBitField64::value_type>(d_field.Bits().data()),
      thrust::device_ptr<LBitField64::value_type>(d_field.Bits().data() + d_field.Bits().size()),
      h_field_storage.data());
  LBitField64 h_field{{h_field_storage.data(), h_field_storage.data() + h_field_storage.size()}};

  for (size_t i = 0; i < h_field.Capacity(); ++i) {
    if (positions.find(i) != positions.cend()) {
      ASSERT_TRUE(h_field.Check(i));
    } else {
      ASSERT_FALSE(h_field.Check(i));
    }
  }
}

}  // anonymous namespace

TEST(GPUFeatureInteractionConstraint, Init) {
  {
    int32_t constexpr kFeatures = 6;
    tree::TrainParam param = GetParameter();
    FConstraintWrapper constraints(param, kFeatures);
    ASSERT_EQ(constraints.Features(), kFeatures);
    common::Span<LBitField64> s_nodes_constraints = constraints.GetNodeConstraints();
    for (LBitField64 const& d_node : s_nodes_constraints) {
      std::vector<LBitField64::value_type> h_node_storage(d_node.Bits().size());
      thrust::copy(thrust::device_ptr<LBitField64::value_type const>(d_node.Bits().data()),
                   thrust::device_ptr<LBitField64::value_type const>(d_node.Bits().data() +
                                                                     d_node.Bits().size()),
                   h_node_storage.data());
      LBitField64 h_node{{h_node_storage.data(), h_node_storage.data() + h_node_storage.size()}};
      // no feature is attached to node.
      for (size_t i = 0; i < h_node.Capacity(); ++i) {
        ASSERT_FALSE(h_node.Check(i));
      }
    }
  }

  {
    // Test one feature in multiple sets
    int32_t constexpr kFeatures = 7;
    tree::TrainParam param = GetParameter();
    param.interaction_constraints = R"([[0, 1, 3], [3, 5, 6]])";
    FConstraintWrapper constraints(param, kFeatures);
    std::vector<bst_feature_t> h_sets{0, 0, 0, 1, 1, 1};
    std::vector<size_t> h_sets_ptr{0, 1, 2, 2, 4, 4, 5, 6};
    auto d_sets = constraints.GetDSets();
    ASSERT_EQ(h_sets.size(), d_sets.size());
    auto d_sets_ptr = constraints.GetSetsPtr();
    ASSERT_EQ(h_sets_ptr, d_sets_ptr);
    for (size_t i = 0; i < h_sets.size(); ++i) {
      ASSERT_EQ(h_sets[i], d_sets[i]);
    }
    for (size_t i = 0; i < h_sets_ptr.size(); ++i) {
      ASSERT_EQ(h_sets_ptr[i], d_sets_ptr[i]);
    }
  }

  {
    // Test having more than 1 LBitField64::value_type
    int32_t constexpr kFeatures = 129;
    tree::TrainParam param = GetParameter();
    param.interaction_constraints = R"([[0, 1, 3], [3, 5, 128], [127, 128]])";
    FConstraintWrapper constraints(param, kFeatures);
    auto d_sets = constraints.GetDSets();
    auto d_sets_ptr = constraints.GetSetsPtr();
    auto _128_beg = d_sets_ptr[128];
    auto _128_end = d_sets_ptr[128 + 1];
    ASSERT_EQ(_128_end - _128_beg, 2);
    ASSERT_EQ(d_sets[_128_beg], 1);
    ASSERT_EQ(d_sets[_128_end - 1], 2);
  }
}

TEST(GPUFeatureInteractionConstraint, Split) {
  auto ctx = MakeCUDACtx(0);
  tree::TrainParam param = GetParameter();
  int32_t constexpr kFeatures = 6;
  FConstraintWrapper constraints(param, kFeatures);

  constraints.Split(&ctx, 0, /*feature_id=*/1, 1, 2);
  for (auto nid : {1, 2}) {
    CompareBitField(constraints.GetNodeConstraints()[nid], {1, 2});
  }
  constraints.Split(&ctx, 1, /*feature_id=*/2, 3, 4);
  for (auto nid : {2, 3, 4}) {
    CompareBitField(constraints.GetNodeConstraints()[nid], {1, 2});
  }
}

TEST(GPUFeatureInteractionConstraint, QueryNode) {
  auto ctx = MakeCUDACtx(0);
  tree::TrainParam param = GetParameter();
  bst_feature_t constexpr kFeatures = 6;
  FConstraintWrapper constraints(param, kFeatures);
  auto out = constraints.PrepareBatch(kFeatures);

  {
    auto span = constraints.QueryNode(&ctx, 0, out);
    ASSERT_EQ(span.size(), kFeatures);
  }

  {
    constraints.Split(&ctx, /*node_id=*/0, /*feature_id=*/1, 1, 2);
    auto span = constraints.QueryNode(&ctx, 1, out);
    std::vector<bst_feature_t> h_result(span.size());
    thrust::copy(thrust::device_ptr<bst_feature_t>(span.data()),
                 thrust::device_ptr<bst_feature_t>(span.data() + span.size()), h_result.begin());
    ASSERT_EQ(h_result.size(), 2);
    ASSERT_EQ(h_result[0], 1);
    ASSERT_EQ(h_result[1], 2);
  }

  {
    tree::TrainParam large_param = GetParameter();
    large_param.interaction_constraints = R"([[1, 139], [244, 0], [139, 221]])";
    FConstraintWrapper large_features(large_param, 256);
    large_features.Split(&ctx, 0, 139, 1, 2);
    auto large_out = large_features.PrepareBatch(large_features.Features());
    auto span = large_features.QueryNode(&ctx, 2, large_out);
    std::vector<bst_feature_t> h_result(span.size());
    thrust::copy(thrust::device_ptr<bst_feature_t>(span.data()),
                 thrust::device_ptr<bst_feature_t>(span.data() + span.size()), h_result.begin());
    ASSERT_EQ(h_result.size(), 3);
    ASSERT_EQ(h_result[0], 1);
    ASSERT_EQ(h_result[1], 139);
    ASSERT_EQ(h_result[2], 221);
  }
}

namespace {
void CompareFeatureList(common::Span<bst_feature_t const> s_output,
                        std::vector<bst_feature_t> solution) {
  std::vector<bst_feature_t> h_output(s_output.size());
  thrust::copy(dh::tcbegin(s_output), dh::tcend(s_output), h_output.begin());
  ASSERT_EQ(h_output.size(), solution.size());
  for (size_t i = 0; i < solution.size(); ++i) {
    ASSERT_EQ(h_output[i], solution[i]);
  }
}
}  // anonymous namespace

TEST(GPUFeatureInteractionConstraint, Query) {
  auto ctx = MakeCUDACtx(0);
  {
    tree::TrainParam param = GetParameter();
    bst_feature_t constexpr kFeatures = 6;
    FConstraintWrapper constraints(param, kFeatures);
    std::vector<bst_feature_t> h_input_feature_list{0, 1, 2, 3, 4, 5};
    dh::device_vector<bst_feature_t> d_input_feature_list(h_input_feature_list);
    common::Span<bst_feature_t> s_input_feature_list = dh::ToSpan(d_input_feature_list);

    auto s_output = constraints.Query(&ctx, s_input_feature_list, 0, {});
    CompareFeatureList(s_output, h_input_feature_list);
  }
  {
    tree::TrainParam param = GetParameter();
    bst_feature_t constexpr kFeatures = 6;
    FConstraintWrapper constraints(param, kFeatures);
    constraints.Split(&ctx, /*node_id=*/0, /*feature_id=*/1, /*left_id=*/1, /*right_id=*/2);
    dh::device_vector<bst_feature_t> sampled{0, 1, 2, 3};
    auto features = dh::ToSpan(sampled);
    auto batch = constraints.PrepareBatch(4);
    auto left = constraints.Query(&ctx, features.first(2), 1, batch.first(2));
    auto right = constraints.Query(&ctx, features.subspan(2), 2, batch.subspan(2));
    CompareFeatureList(left, {1});
    CompareFeatureList(right, {2});
  }

  // Test shared feature
  {
    tree::TrainParam param = GetParameter();
    bst_feature_t constexpr kFeatures = 6;
    std::string const constraints_str = R"constraint([[1, 2], [2, 3, 4]])constraint";
    param.interaction_constraints = constraints_str;

    FConstraintWrapper constraints(param, kFeatures);
    constraints.Split(&ctx, /*node_id=*/0, /*feature_id=*/2, /*left_id=*/1, /*right_id=*/2);

    std::vector<bst_feature_t> h_input_feature_list{0, 1, 2, 3, 4, 5};
    dh::device_vector<bst_feature_t> d_input_feature_list(h_input_feature_list);
    common::Span<bst_feature_t> s_input_feature_list = dh::ToSpan(d_input_feature_list);

    auto out = constraints.PrepareBatch(kFeatures);
    auto s_output = constraints.Query(&ctx, s_input_feature_list, 1, out);
    CompareFeatureList(s_output, {1, 2, 3, 4});
  }

  // Test choosing free feature in root
  {
    tree::TrainParam param = GetParameter();
    bst_feature_t constexpr kFeatures = 6;
    std::string const constraints_str = R"constraint([[0, 1]])constraint";
    param.interaction_constraints = constraints_str;
    FConstraintWrapper constraints(param, kFeatures);
    std::vector<bst_feature_t> h_input_feature_list{0, 1, 2, 3, 4, 5};
    dh::device_vector<bst_feature_t> d_input_feature_list(h_input_feature_list);
    common::Span<bst_feature_t> s_input_feature_list = dh::ToSpan(d_input_feature_list);
    constraints.Split(&ctx, /*node_id=*/0, /*feature_id=*/2, /*left_id=*/1, /*right_id=*/2);
    auto out = constraints.PrepareBatch(kFeatures);
    auto s_output = constraints.Query(&ctx, s_input_feature_list, 1, out);
    CompareFeatureList(s_output, {2});
    s_output = constraints.Query(&ctx, s_input_feature_list, 2, out);
    CompareFeatureList(s_output, {2});
  }
}

// Compare eligible features after each prescribed split. The sibling is queried again
// after its peer splits, since the two branches can retain different groups.
TEST(GPUFeatureInteractionConstraint, PathMatchesCPU) {
  auto ctx = MakeCUDACtx(0);
  bst_feature_t constexpr kFeatures = 5;
  dh::device_vector<bst_feature_t> sampled{0, 1, 2, 3, 4};
  auto check_paths = [&](std::string const& groups, bst_feature_t root_feature,
                         bst_feature_t left_feature, bst_feature_t right_feature) {
    auto param = GetParameter();
    param.interaction_constraints = groups;
    FeatureInteractionConstraintHost cpu;
    cpu.Configure(param, kFeatures);
    FeatureInteractionConstraintDevice gpu{param, kFeatures};
    auto out = gpu.PrepareBatch(kFeatures);
    auto check = [&](bst_node_t nid) {
      std::vector<bst_feature_t> expected;
      for (bst_feature_t f = 0; f < kFeatures; ++f) {
        if (cpu.Query(nid, f)) {
          expected.push_back(f);
        }
      }
      CompareFeatureList(gpu.Query(&ctx, dh::ToSpan(sampled), nid, out), expected);
    };
    auto split = [&](bst_node_t nid, bst_feature_t f, bst_node_t left, bst_node_t right) {
      ASSERT_TRUE(cpu.Query(nid, f));
      cpu.Split(nid, f, left, right);
      gpu.Split(&ctx, nid, f, left, right);
      check(left);
      check(right);
    };
    check(0);
    split(0, root_feature, 1, 2);
    split(1, left_feature, 3, 4);
    check(2);
    split(2, right_feature, 5, 6);
    check(3);
  };

  check_paths("[[0,1],[1,3,4]]", 0, 1, 0);
  check_paths("[[0,1],[1,3,4]]", 1, 0, 4);
  check_paths("[[0,1],[0,2],[1,2]]", 0, 1, 2);
  check_paths("[[0,1]]", 4, 4, 4);
}

}  // namespace xgboost
