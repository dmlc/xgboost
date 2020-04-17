/*!
 * Copyright 2019 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <cinttypes>
#include <string>
#include <bitset>
#include <set>
#include "../../../src/tree/constraints.cuh"
#include "../../../src/tree/param.h"
#include "../../../src/common/device_helpers.cuh"

namespace xgboost {
namespace {

struct FConstraintWrapper : public FeatureInteractionConstraintDevice {
  common::Span<LBitField64> GetNodeConstraints() {
    return FeatureInteractionConstraintDevice::s_node_constraints_;
  }
  FConstraintWrapper(tree::TrainParam param, bst_feature_t n_features) :
      FeatureInteractionConstraintDevice(param, n_features) {}

  dh::device_vector<bst_feature_t> const& GetDSets() const {
    return d_sets_;
  }
  dh::device_vector<size_t> const& GetDSetsPtr() const {
    return d_sets_ptr_;
  }
};

std::string GetConstraintsStr() {
  std::string const constraints_str = R"constraint([[1, 2], [3, 4, 5]])constraint";
  return constraints_str;
}

tree::TrainParam GetParameter() {
  std::vector<std::pair<std::string, std::string>> args{
    {"interaction_constraints", GetConstraintsStr()}
  };
  tree::TrainParam param;
  param.Init(args);
  return param;
}

void CompareBitField(LBitField64 d_field, std::set<uint32_t> positions) {
  std::vector<LBitField64::value_type> h_field_storage(d_field.Bits().size());
  thrust::copy(thrust::device_ptr<LBitField64::value_type>(d_field.Bits().data()),
               thrust::device_ptr<LBitField64::value_type>(
                   d_field.Bits().data() + d_field.Bits().size()),
               h_field_storage.data());
  LBitField64 h_field{ {h_field_storage.data(),
                        h_field_storage.data() + h_field_storage.size()} };

  for (size_t i = 0; i < h_field.Size(); ++i) {
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
                   thrust::device_ptr<LBitField64::value_type const>(
                       d_node.Bits().data() + d_node.Bits().size()),
                   h_node_storage.data());
      LBitField64 h_node {
        {h_node_storage.data(), h_node_storage.data() +  h_node_storage.size()}
      };
      // no feature is attached to node.
      for (size_t i = 0; i < h_node.Size(); ++i) {
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
    std::vector<int32_t> h_sets {0, 0, 0, 1, 1, 1};
    std::vector<int32_t> h_sets_ptr {0, 1, 2, 2, 4, 4, 5, 6};
    auto d_sets = constraints.GetDSets();
    ASSERT_EQ(h_sets.size(), d_sets.size());
    auto d_sets_ptr = constraints.GetDSetsPtr();
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
    auto d_sets_ptr = constraints.GetDSetsPtr();
    auto _128_beg = d_sets_ptr[128];
    auto _128_end = d_sets_ptr[128 + 1];
    ASSERT_EQ(_128_end - _128_beg, 2);
    ASSERT_EQ(d_sets[_128_beg], 1);
    ASSERT_EQ(d_sets[_128_end-1], 2);
  }
}

TEST(GPUFeatureInteractionConstraint, Split) {
  tree::TrainParam param = GetParameter();
  int32_t constexpr kFeatures = 6;
  FConstraintWrapper constraints(param, kFeatures);

  {
    LBitField64 d_node[3];
    constraints.Split(0, /*feature_id=*/1, 1, 2);
    for (size_t nid = 0; nid < 3; ++nid) {
      d_node[nid] = constraints.GetNodeConstraints()[nid];
      ASSERT_EQ(d_node[nid].Bits().size(), 1);
      CompareBitField(d_node[nid], {1, 2});
    }
  }

  {
    LBitField64 d_node[5];
    constraints.Split(1, /*feature_id=*/0, /*left_id=*/3, /*right_id=*/4);
    for (auto nid : {1, 3, 4}) {
      d_node[nid] = constraints.GetNodeConstraints()[nid];
      CompareBitField(d_node[nid], {0, 1, 2});
    }
    for (auto nid : {0, 2}) {
      d_node[nid] = constraints.GetNodeConstraints()[nid];
      CompareBitField(d_node[nid], {1, 2});
    }
  }
}

TEST(GPUFeatureInteractionConstraint, QueryNode) {
  tree::TrainParam param = GetParameter();
  bst_feature_t constexpr kFeatures = 6;
  FConstraintWrapper constraints(param, kFeatures);

  {
    auto span = constraints.QueryNode(0);
    ASSERT_EQ(span.size(), 0);
  }

  {
    constraints.Split(/*node_id=*/ 0, /*feature_id=*/ 1, 1, 2);
    auto span = constraints.QueryNode(0);
    std::vector<bst_feature_t> h_result (span.size());
    thrust::copy(thrust::device_ptr<bst_feature_t>(span.data()),
                 thrust::device_ptr<bst_feature_t>(span.data() + span.size()),
                 h_result.begin());
    ASSERT_EQ(h_result.size(), 2);
    ASSERT_EQ(h_result[0], 1);
    ASSERT_EQ(h_result[1], 2);
  }

  {
    constraints.Split(1, /*feature_id=*/0, 3, 4);
    auto span = constraints.QueryNode(1);
    std::vector<bst_feature_t> h_result (span.size());
    thrust::copy(thrust::device_ptr<bst_feature_t>(span.data()),
                 thrust::device_ptr<bst_feature_t>(span.data() + span.size()),
                 h_result.begin());
    ASSERT_EQ(h_result.size(), 3);
    ASSERT_EQ(h_result[0], 0);
    ASSERT_EQ(h_result[1], 1);
    ASSERT_EQ(h_result[2], 2);

    // same as parent
    span = constraints.QueryNode(3);
    h_result.resize(span.size());
    thrust::copy(thrust::device_ptr<bst_feature_t>(span.data()),
                 thrust::device_ptr<bst_feature_t>(span.data() + span.size()),
                 h_result.begin());
    ASSERT_EQ(h_result.size(), 3);
    ASSERT_EQ(h_result[0], 0);
    ASSERT_EQ(h_result[1], 1);
    ASSERT_EQ(h_result[2], 2);
  }

  {
    tree::TrainParam large_param = GetParameter();
    large_param.interaction_constraints = R"([[1, 139], [244, 0], [139, 221]])";
    FConstraintWrapper large_features(large_param, 256);
    large_features.Split(0, 139, 1, 2);
    auto span = large_features.QueryNode(0);
    std::vector<bst_feature_t> h_result (span.size());
    thrust::copy(thrust::device_ptr<bst_feature_t>(span.data()),
                 thrust::device_ptr<bst_feature_t>(span.data() + span.size()),
                 h_result.begin());
    ASSERT_EQ(h_result.size(), 3);
    ASSERT_EQ(h_result[0], 1);
    ASSERT_EQ(h_result[1], 139);
    ASSERT_EQ(h_result[2], 221);
  }
}

namespace {

void CompareFeatureList(common::Span<bst_feature_t> s_output, std::vector<bst_feature_t> solution) {
  std::vector<bst_feature_t> h_output(s_output.size());
  thrust::copy(thrust::device_ptr<bst_feature_t>(s_output.data()),
               thrust::device_ptr<bst_feature_t>(s_output.data() + s_output.size()),
               h_output.begin());
  ASSERT_EQ(h_output.size(), solution.size());
  for (size_t i = 0; i < solution.size(); ++i) {
    ASSERT_EQ(h_output[i], solution[i]);
  }
}

}  // anonymous namespace

TEST(GPUFeatureInteractionConstraint, Query) {
  {
    tree::TrainParam param = GetParameter();
    bst_feature_t constexpr kFeatures = 6;
    FConstraintWrapper constraints(param, kFeatures);
    std::vector<bst_feature_t> h_input_feature_list {0, 1, 2, 3, 4, 5};
    dh::device_vector<bst_feature_t> d_input_feature_list (h_input_feature_list);
    common::Span<bst_feature_t> s_input_feature_list = dh::ToSpan(d_input_feature_list);

    auto s_output = constraints.Query(s_input_feature_list, 0);
    CompareFeatureList(s_output, h_input_feature_list);
  }
  {
    tree::TrainParam param = GetParameter();
    bst_feature_t constexpr kFeatures = 6;
    FConstraintWrapper constraints(param, kFeatures);
    constraints.Split(/*node_id=*/0, /*feature_id=*/1, /*left_id=*/1, /*right_id=*/2);
    constraints.Split(/*node_id=*/1, /*feature_id=*/0, /*left_id=*/3, /*right_id=*/4);
    constraints.Split(/*node_id=*/4, /*feature_id=*/3, /*left_id=*/5, /*right_id=*/6);
    /*
     * (node id) [allowed features]
     *
     *               (0) [1, 2]
     *           /        \
     *      {split at 0}   \
     *         /            \
     *        (1)[0, 1, 2]  (2)[1, 2]
     *     /        \
     *    /      {split at 3}
     *   /            \
     * (3)[0, 1, 2]   (4)[0, 1, 2, 3, 4, 5]
     *
     */

    std::vector<bst_feature_t> h_input_feature_list {0, 1, 2, 3, 4, 5};
    dh::device_vector<bst_feature_t> d_input_feature_list (h_input_feature_list);
    common::Span<bst_feature_t> s_input_feature_list = dh::ToSpan(d_input_feature_list);

    auto s_output = constraints.Query(s_input_feature_list, 1);
    CompareFeatureList(s_output, {0, 1, 2});
    s_output = constraints.Query(s_input_feature_list, 2);
    CompareFeatureList(s_output, {1, 2});
    s_output = constraints.Query(s_input_feature_list, 3);
    CompareFeatureList(s_output, {0, 1, 2});
    s_output = constraints.Query(s_input_feature_list, 4);
    CompareFeatureList(s_output, {0, 1, 2, 3, 4, 5});
    s_output = constraints.Query(s_input_feature_list, 5);
    CompareFeatureList(s_output, {0, 1, 2, 3, 4, 5});
    s_output = constraints.Query(s_input_feature_list, 6);
    CompareFeatureList(s_output, {0, 1, 2, 3, 4, 5});
  }

  // Test shared feature
  {
    tree::TrainParam param = GetParameter();
    bst_feature_t constexpr kFeatures = 6;
    std::string const constraints_str = R"constraint([[1, 2], [2, 3, 4]])constraint";
    param.interaction_constraints = constraints_str;

    FConstraintWrapper constraints(param, kFeatures);
    constraints.Split(/*node_id=*/0, /*feature_id=*/2, /*left_id=*/1, /*right_id=*/2);

    std::vector<bst_feature_t> h_input_feature_list {0, 1, 2, 3, 4, 5};
    dh::device_vector<bst_feature_t> d_input_feature_list (h_input_feature_list);
    common::Span<bst_feature_t> s_input_feature_list = dh::ToSpan(d_input_feature_list);

    auto s_output = constraints.Query(s_input_feature_list, 1);
    CompareFeatureList(s_output, {1, 2, 3, 4});
  }

  // Test choosing free feature in root
  {
    tree::TrainParam param = GetParameter();
    bst_feature_t constexpr kFeatures = 6;
    std::string const constraints_str = R"constraint([[0, 1]])constraint";
    param.interaction_constraints = constraints_str;
    FConstraintWrapper constraints(param, kFeatures);
    std::vector<bst_feature_t> h_input_feature_list {0, 1, 2, 3, 4, 5};
    dh::device_vector<bst_feature_t> d_input_feature_list (h_input_feature_list);
    common::Span<bst_feature_t> s_input_feature_list = dh::ToSpan(d_input_feature_list);
    constraints.Split(/*node_id=*/0, /*feature_id=*/2, /*left_id=*/1, /*right_id=*/2);
    auto s_output = constraints.Query(s_input_feature_list, 1);
    CompareFeatureList(s_output, {2});
    s_output = constraints.Query(s_input_feature_list, 2);
    CompareFeatureList(s_output, {2});
  }
}

}  // namespace xgboost
