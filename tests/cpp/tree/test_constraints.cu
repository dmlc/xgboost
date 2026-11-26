/**
 * Copyright 2019-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

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
  using FeatureInteractionConstraintDevice::FeatureMasks;
  dh::device_vector<GroupId> const& GetDSets() const { return state_->groups; }
  std::size_t ResultCapacity() const { return state_->result_buffer.Capacity(); }
  std::vector<size_t> const& GetSetsPtr() const { return state_->feature_ptr; }
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
  auto ctx = MakeCUDACtx(0);
  {
    bst_feature_t constexpr kFeatures = 6;
    tree::TrainParam param = GetParameter();
    FConstraintWrapper constraints;
    constraints.Reset(&ctx, param, kFeatures);
    ASSERT_EQ(constraints.Features(), kFeatures);
    auto masks = constraints.FeatureMasks();
    for (bst_node_t nidx = 0; nidx < param.MaxNodes(); ++nidx) {
      CompareBitField(masks[nidx], {});
    }
  }

  {
    // Test one feature in multiple sets
    int32_t constexpr kFeatures = 7;
    tree::TrainParam param = GetParameter();
    param.interaction_constraints = R"([[0, 1, 3], [3, 5, 6]])";
    FConstraintWrapper constraints;
    constraints.Reset(&ctx, param, kFeatures);
    std::vector<FConstraintWrapper::GroupId> h_sets{0, 0, 0, 1, 1, 1};
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
    FConstraintWrapper constraints;
    constraints.Reset(&ctx, param, kFeatures);
    auto d_sets = constraints.GetDSets();
    auto d_sets_ptr = constraints.GetSetsPtr();
    auto _128_beg = d_sets_ptr[128];
    auto _128_end = d_sets_ptr[128 + 1];
    ASSERT_EQ(_128_end - _128_beg, 2);
    ASSERT_EQ(d_sets[_128_beg], 1);
    ASSERT_EQ(d_sets[_128_end - 1], 2);
    constraints.Split(&ctx, 0, 128, 1, 2);
    CompareBitField(constraints.FeatureMasks()[1], {3, 5, 127, 128});
    constraints.Reset(&ctx, param, kFeatures);
    CompareBitField(constraints.FeatureMasks()[1], {});
  }
}

TEST(GPUFeatureInteractionConstraint, Split) {
  auto ctx = MakeCUDACtx(0);
  tree::TrainParam param = GetParameter();
  int32_t constexpr kFeatures = 6;
  FConstraintWrapper constraints;
  constraints.Reset(&ctx, param, kFeatures);

  constraints.Split(&ctx, 0, /*feature_id=*/1, 1, 2);
  for (auto nid : {1, 2}) {
    CompareBitField(constraints.FeatureMasks()[nid], {1, 2});
  }
  constraints.Split(&ctx, 1, /*feature_id=*/2, 3, 4);
  for (auto nid : {2, 3, 4}) {
    CompareBitField(constraints.FeatureMasks()[nid], {1, 2});
  }
}

namespace {
struct NodeFeatures {
  bst_node_t nidx;
  common::Span<bst_feature_t const> feature_set;
};

bst_feature_t QueryBatch(Context const* ctx, FeatureInteractionConstraintDevice* constraints,
                         common::Span<NodeFeatures> inputs) {
  dh::DeviceUVector<NodeFeatures> d_inputs(inputs.size());
  auto max_active = constraints->Query(ctx, inputs, dh::ToSpan(d_inputs));
  std::vector<NodeFeatures> evaluated;
  dh::CopyTo(d_inputs, &evaluated, ctx->CUDACtx()->Stream());
  ctx->CUDACtx()->Stream().Sync();
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(evaluated[i].nidx, inputs[i].nidx);
    EXPECT_EQ(evaluated[i].feature_set.data(), inputs[i].feature_set.data());
    EXPECT_EQ(evaluated[i].feature_set.size(), inputs[i].feature_set.size());
  }
  return max_active;
}

auto QueryNodeFeatures(Context const* ctx, FeatureInteractionConstraintDevice* constraints,
                       common::Span<bst_feature_t const> sampled, bst_node_t nidx) {
  NodeFeatures input{nidx, sampled};
  QueryBatch(ctx, constraints, common::Span{&input, 1});
  return input.feature_set;
}

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
    FConstraintWrapper constraints;
    constraints.Reset(&ctx, param, kFeatures);
    std::vector<bst_feature_t> h_input_feature_list{0, 1, 2, 3, 4, 5};
    dh::device_vector<bst_feature_t> d_input_feature_list(h_input_feature_list);
    common::Span<bst_feature_t> s_input_feature_list = dh::ToSpan(d_input_feature_list);

    auto s_output = QueryNodeFeatures(&ctx, &constraints, s_input_feature_list, 0);
    CompareFeatureList(s_output, h_input_feature_list);
  }
  {
    tree::TrainParam param = GetParameter();
    bst_feature_t constexpr kFeatures = 6;
    FConstraintWrapper constraints;
    constraints.Reset(&ctx, param, kFeatures);
    constraints.Split(&ctx, /*node_id=*/0, /*feature_id=*/1, /*left_id=*/1, /*right_id=*/2);
    dh::device_vector<bst_feature_t> sampled{0, 1, 2, 3};
    auto features = dh::ToSpan(sampled);
    NodeFeatures empty[]{{1, features.first(1)}, {2, features.first(1)}};
    ASSERT_EQ(QueryBatch(&ctx, &constraints, common::Span<NodeFeatures>{empty}), 0);
    ASSERT_TRUE(empty[0].feature_set.empty());
    ASSERT_TRUE(empty[1].feature_set.empty());
    ASSERT_EQ(constraints.ResultCapacity(), 0);

    NodeFeatures batch[]{{1, features.first(2)},
                         {2, features.subspan(2)},
                         {1, features.subspan(1, 2)},
                         {2, features.first(1)}};
    ASSERT_EQ(QueryBatch(&ctx, &constraints, common::Span<NodeFeatures>{batch}), 2);
    CompareFeatureList(batch[0].feature_set, {1});
    CompareFeatureList(batch[1].feature_set, {2});
    ASSERT_EQ(batch[2].feature_set.data(), features.data() + 1);
    ASSERT_TRUE(batch[3].feature_set.empty());
    ASSERT_EQ(constraints.ResultCapacity(), 2);
  }

  // An overlapping group and a sampled list spanning multiple compaction tiles.
  {
    auto param = GetParameter();
    param.interaction_constraints = R"([[1, 2, 510], [2, 4, 512, 1026]])";
    FConstraintWrapper constraints;
    constraints.Reset(&ctx, param, 1027);
    constraints.Split(&ctx, 0, 2, 1, 2);

    dh::device_vector<bst_feature_t> sampled(514);
    thrust::sequence(sampled.begin(), sampled.end(), 0, 2);
    auto output = QueryNodeFeatures(&ctx, &constraints, dh::ToSpan(sampled), 1);
    CompareFeatureList(output, {2, 4, 510, 512, 1026});
  }

  // Test choosing free feature in root
  {
    tree::TrainParam param = GetParameter();
    bst_feature_t constexpr kFeatures = 6;
    std::string const constraints_str = R"constraint([[0, 1]])constraint";
    param.interaction_constraints = constraints_str;
    FConstraintWrapper constraints;
    constraints.Reset(&ctx, param, kFeatures);
    std::vector<bst_feature_t> h_input_feature_list{0, 1, 2, 3, 4, 5};
    dh::device_vector<bst_feature_t> d_input_feature_list(h_input_feature_list);
    common::Span<bst_feature_t> s_input_feature_list = dh::ToSpan(d_input_feature_list);
    constraints.Split(&ctx, /*node_id=*/0, /*feature_id=*/2, /*left_id=*/1, /*right_id=*/2);
    auto s_output = QueryNodeFeatures(&ctx, &constraints, s_input_feature_list, 1);
    CompareFeatureList(s_output, {2});
    s_output = QueryNodeFeatures(&ctx, &constraints, s_input_feature_list, 2);
    CompareFeatureList(s_output, {2});
  }
}

TEST(GPUFeatureInteractionConstraint, WideBatch) {
  auto ctx = MakeCUDACtx(0);
  auto param = GetParameter();
  param.max_depth = 10;
  bst_node_t constexpr kChildren = 1 << 10;
  bst_feature_t constexpr kFeatures = 100000;
  FConstraintWrapper constraints;
  constraints.Reset(&ctx, param, kFeatures);
  // Feature 0 belongs to no group, so every descendant can only split on 0.
  for (bst_node_t nidx = 0; nidx < kChildren - 1; ++nidx) {
    constraints.Split(&ctx, nidx, 0, 2 * nidx + 1, 2 * nidx + 2);
  }
  dh::device_vector<bst_feature_t> features(kFeatures);
  thrust::sequence(features.begin(), features.end());
  std::vector<NodeFeatures> inputs;
  for (bst_node_t i = 0; i < kChildren; ++i) {
    inputs.push_back({kChildren - 1 + i, dh::ToSpan(features)});
  }
  ASSERT_EQ(QueryBatch(&ctx, &constraints, common::Span{inputs}), 1);
  ASSERT_EQ(constraints.ResultCapacity(), kChildren);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    ASSERT_EQ(inputs[i].feature_set.size(), 1);
    ASSERT_EQ(inputs[i].feature_set.data(), inputs.front().feature_set.data() + i);
  }
  CompareFeatureList({inputs.front().feature_set.data(), inputs.size()},
                     std::vector<bst_feature_t>(kChildren, 0));
}

// Compare eligible features after each prescribed split. The sibling is queried again
// after its peer splits, since the two branches can retain different groups.
TEST(GPUFeatureInteractionConstraint, PathMatchesCPU) {
  auto ctx = MakeCUDACtx(0);
  bst_feature_t constexpr kFeatures = 5;
  dh::device_vector<bst_feature_t> sampled{0, 1, 2, 3, 4};
  FeatureInteractionConstraintDevice gpu;
  auto check_paths = [&](std::string const& groups, bst_feature_t root_feature,
                         bst_feature_t left_feature, bst_feature_t right_feature) {
    auto param = GetParameter();
    param.interaction_constraints = groups;
    FeatureInteractionConstraintHost cpu;
    cpu.Configure(param, kFeatures);
    gpu.Reset(&ctx, param, kFeatures);
    auto check = [&](bst_node_t nid) {
      std::vector<bst_feature_t> expected;
      for (bst_feature_t f = 0; f < kFeatures; ++f) {
        if (cpu.Query(nid, f)) {
          expected.push_back(f);
        }
      }
      CompareFeatureList(QueryNodeFeatures(&ctx, &gpu, dh::ToSpan(sampled), nid), expected);
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
