/**
 * Copyright 2020-2024 by XGBoost contributors
 */
#include <gtest/gtest.h>

#include <oneapi/dpl/random>

#include "../../../plugin/sycl/tree/hist_updater.h"
#include "../../../plugin/sycl/device_manager.h"

#include "../../../src/tree/common_row_partitioner.h"

#include "../helpers.h"

namespace xgboost::sycl::tree {

// Use this class to test the protected methods of HistUpdater
template <typename GradientSumT>
class TestHistUpdater : public HistUpdater<GradientSumT> {
 public:
  TestHistUpdater(const Context* ctx,
                  ::sycl::queue* qu,
                  const xgboost::tree::TrainParam& param,
                  FeatureInteractionConstraintHost int_constraints_,
                  DMatrix const* fmat) : HistUpdater<GradientSumT>(ctx, qu, param,
                                                                   int_constraints_, fmat) {}

  void TestInitSampling(const HostDeviceVector<GradientPair>& gpair,
                        USMVector<size_t, MemoryType::on_device>* row_indices) {
    HistUpdater<GradientSumT>::InitSampling(gpair, row_indices);
  }

  auto* TestInitData(const common::GHistIndexMatrix& gmat,
                     const HostDeviceVector<GradientPair>& gpair,
                     const DMatrix& fmat,
                     const RegTree& tree) {
    HistUpdater<GradientSumT>::InitData(gmat, gpair, fmat, tree);
    return &(HistUpdater<GradientSumT>::row_set_collection_);
  }

  const auto* TestBuildHistogramsLossGuide(ExpandEntry entry,
                                    const common::GHistIndexMatrix &gmat,
                                    RegTree *p_tree,
                                    const HostDeviceVector<GradientPair>& gpair) {
    HistUpdater<GradientSumT>::BuildHistogramsLossGuide(entry, gmat, p_tree, gpair);
    return &(HistUpdater<GradientSumT>::hist_);
  }

  auto TestInitNewNode(int nid,
                       const common::GHistIndexMatrix& gmat,
                       const HostDeviceVector<GradientPair>& gpair,
                       const RegTree& tree) {
    HistUpdater<GradientSumT>::InitNewNode(nid, gmat, gpair, tree);
    return HistUpdater<GradientSumT>::snode_host_[nid];
  }

  auto TestEvaluateSplits(const std::vector<ExpandEntry>& nodes_set,
                          const common::GHistIndexMatrix& gmat,
                          const RegTree& tree) {
    HistUpdater<GradientSumT>::EvaluateSplits(nodes_set, gmat, tree);
    return HistUpdater<GradientSumT>::snode_host_;
  }

  void TestApplySplit(const std::vector<ExpandEntry> nodes,
                      const common::GHistIndexMatrix& gmat,
                      RegTree* p_tree) {
    HistUpdater<GradientSumT>::ApplySplit(nodes, gmat, p_tree);
  }

  auto TestExpandWithLossGuide(const common::GHistIndexMatrix& gmat,
                               DMatrix *p_fmat,
                               RegTree* p_tree,
                               const HostDeviceVector<GradientPair>& gpair) {
    HistUpdater<GradientSumT>::ExpandWithLossGuide(gmat, p_tree, gpair);
  }

  auto TestExpandWithDepthWise(const common::GHistIndexMatrix& gmat,
                               DMatrix *p_fmat,
                               RegTree* p_tree,
                               const HostDeviceVector<GradientPair>& gpair) {
    HistUpdater<GradientSumT>::ExpandWithDepthWise(gmat, p_tree, gpair);
  }
};

void GenerateRandomGPairs(::sycl::queue* qu, GradientPair* gpair_ptr, size_t num_rows, bool has_neg_hess) {
  qu->submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(::sycl::range<1>(num_rows)),
                                        [=](::sycl::item<1> pid) {
      uint64_t i = pid.get_linear_id();

      constexpr uint32_t seed = 777;
      oneapi::dpl::minstd_rand engine(seed, i);
      GradientPair::ValueT smallest_hess_val = has_neg_hess ? -1. : 0.;
      oneapi::dpl::uniform_real_distribution<GradientPair::ValueT> distr(smallest_hess_val, 1.);
      gpair_ptr[i] = {distr(engine), distr(engine)};
    });
  });
  qu->wait();
}

template <typename GradientSumT>
void TestHistUpdaterSampling(const xgboost::tree::TrainParam& param) {
  const size_t num_rows = 1u << 12;
  const size_t num_columns = 1;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  auto p_fmat = RandomDataGenerator{num_rows, num_columns, 0.0}.GenerateDMatrix();

  FeatureInteractionConstraintHost int_constraints;

  TestHistUpdater<GradientSumT> updater(&ctx, qu, param, int_constraints, p_fmat.get());

  USMVector<size_t, MemoryType::on_device> row_indices_0(qu, num_rows);
  USMVector<size_t, MemoryType::on_device> row_indices_1(qu, num_rows);
  HostDeviceVector<GradientPair> gpair(num_rows, {0, 0}, ctx.Device());
  GenerateRandomGPairs(qu, gpair.DevicePointer(), num_rows, true);

  updater.TestInitSampling(gpair, &row_indices_0);
  
  size_t n_samples = row_indices_0.Size();
  // Half of gpairs have neg hess
  ASSERT_LT(n_samples, num_rows * 0.5 * param.subsample * 1.2);
  ASSERT_GT(n_samples, num_rows * 0.5 * param.subsample / 1.2);

  // Check if two lanunches generate different realisations:
  updater.TestInitSampling(gpair, &row_indices_1);
  if (row_indices_1.Size() == n_samples) {
    std::vector<size_t> row_indices_0_host(n_samples);
    std::vector<size_t> row_indices_1_host(n_samples);
    qu->memcpy(row_indices_0_host.data(), row_indices_0.Data(), n_samples * sizeof(size_t)).wait();
    qu->memcpy(row_indices_1_host.data(), row_indices_1.Data(), n_samples * sizeof(size_t)).wait();

    // The order in row_indices_0 and row_indices_1 can be different
    std::set<size_t> rows;
    for (auto row : row_indices_0_host) {
      rows.insert(row);
    }

    size_t num_diffs = 0;
    for (auto row : row_indices_1_host) {
      if (rows.count(row) == 0) num_diffs++;
    }

    ASSERT_NE(num_diffs, 0);
  }
}

template <typename GradientSumT>
void TestHistUpdaterInitData(const xgboost::tree::TrainParam& param, bool has_neg_hess) {
  const size_t num_rows = 1u << 8;
  const size_t num_columns = 1;
  const size_t n_bins = 32;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  auto p_fmat = RandomDataGenerator{num_rows, num_columns, 0.0}.GenerateDMatrix();

  FeatureInteractionConstraintHost int_constraints;

  TestHistUpdater<GradientSumT> updater(&ctx, qu, param, int_constraints, p_fmat.get());

  HostDeviceVector<GradientPair> gpair(num_rows, {0, 0}, ctx.Device());
  GenerateRandomGPairs(qu, gpair.DevicePointer(), num_rows, has_neg_hess);

  common::GHistIndexMatrix gmat;
  gmat.Init(qu, &ctx, p_fmat.get(), n_bins);
  RegTree tree;

  auto* row_set_collection = updater.TestInitData(gmat, gpair, *p_fmat, tree);
  auto& row_indices = row_set_collection->Data();

  std::vector<size_t> row_indices_host(row_indices.Size());
  qu->memcpy(row_indices_host.data(), row_indices.DataConst(), row_indices.Size()*sizeof(size_t)).wait();

  if (!has_neg_hess) {
    for (size_t i = 0; i < num_rows; ++i) {
      ASSERT_EQ(row_indices_host[i], i);
    }
  } else {
    std::set<size_t> rows;
    for (size_t i = 0; i < num_rows; ++i) {
      if (gpair.HostVector()[i].GetHess() >= 0.0f) {
        rows.insert(i);
      }
    }
    ASSERT_EQ(rows.size(), row_indices_host.size());
    for (size_t row_idx : row_indices_host) {
      ASSERT_EQ(rows.count(row_idx), 1);
    }
  }
}

template <typename GradientSumT>
void TestHistUpdaterBuildHistogramsLossGuide(const xgboost::tree::TrainParam& param, float sparsity) {
  const size_t num_rows = 1u << 8;
  const size_t num_columns = 1;
  const size_t n_bins = 32;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  auto p_fmat = RandomDataGenerator{num_rows, num_columns, sparsity}.GenerateDMatrix();

  FeatureInteractionConstraintHost int_constraints;

  TestHistUpdater<GradientSumT> updater(&ctx, qu, param, int_constraints, p_fmat.get());
  updater.SetHistSynchronizer(new BatchHistSynchronizer<GradientSumT>());
  updater.SetHistRowsAdder(new BatchHistRowsAdder<GradientSumT>());

  HostDeviceVector<GradientPair> gpair(num_rows, {0, 0}, ctx.Device());
  GenerateRandomGPairs(qu, gpair.DevicePointer(), num_rows, false);

  common::GHistIndexMatrix gmat;
  gmat.Init(qu, &ctx, p_fmat.get(), n_bins);

  RegTree tree;
  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].LeftChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  tree.ExpandNode(tree[0].RightChild(), 0, 0, false, 0, 0, 0, 0, 0, 0, 0);

  ExpandEntry node0(0, tree.GetDepth(0));
  ExpandEntry node1(1, tree.GetDepth(1));
  ExpandEntry node2(2, tree.GetDepth(2));

  auto* row_set_collection = updater.TestInitData(gmat, gpair, *p_fmat, tree);
  row_set_collection->AddSplit(0, 1, 2, 42, num_rows - 42);

  updater.TestBuildHistogramsLossGuide(node0, gmat, &tree, gpair);
  const auto* hist = updater.TestBuildHistogramsLossGuide(node1, gmat, &tree, gpair);

  ASSERT_EQ((*hist)[0].Size(), n_bins);
  ASSERT_EQ((*hist)[1].Size(), n_bins);
  ASSERT_EQ((*hist)[2].Size(), n_bins);

  std::vector<xgboost::detail::GradientPairInternal<GradientSumT>> hist0_host(n_bins);
  std::vector<xgboost::detail::GradientPairInternal<GradientSumT>> hist1_host(n_bins);
  std::vector<xgboost::detail::GradientPairInternal<GradientSumT>> hist2_host(n_bins);
  qu->memcpy(hist0_host.data(), (*hist)[0].DataConst(), sizeof(xgboost::detail::GradientPairInternal<GradientSumT>) * n_bins);
  qu->memcpy(hist1_host.data(), (*hist)[1].DataConst(), sizeof(xgboost::detail::GradientPairInternal<GradientSumT>) * n_bins);
  qu->memcpy(hist2_host.data(), (*hist)[2].DataConst(), sizeof(xgboost::detail::GradientPairInternal<GradientSumT>) * n_bins);
  qu->wait();

  for (size_t idx_bin = 0; idx_bin < n_bins; ++idx_bin) {
    EXPECT_NEAR(hist0_host[idx_bin].GetGrad(), hist1_host[idx_bin].GetGrad() + hist2_host[idx_bin].GetGrad(), 1e-6);
    EXPECT_NEAR(hist0_host[idx_bin].GetHess(), hist1_host[idx_bin].GetHess() + hist2_host[idx_bin].GetHess(), 1e-6);
  }
}

template <typename GradientSumT>
void TestHistUpdaterInitNewNode(const xgboost::tree::TrainParam& param, float sparsity) {
  const size_t num_rows = 1u << 8;
  const size_t num_columns = 1;
  const size_t n_bins = 32;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  auto p_fmat = RandomDataGenerator{num_rows, num_columns, sparsity}.GenerateDMatrix();

  FeatureInteractionConstraintHost int_constraints;

  TestHistUpdater<GradientSumT> updater(&ctx, qu, param, int_constraints, p_fmat.get());
  updater.SetHistSynchronizer(new BatchHistSynchronizer<GradientSumT>());
  updater.SetHistRowsAdder(new BatchHistRowsAdder<GradientSumT>());

  HostDeviceVector<GradientPair> gpair(num_rows, {0, 0}, ctx.Device());
  auto* gpair_ptr = gpair.DevicePointer();
  GenerateRandomGPairs(qu, gpair_ptr, num_rows, false);

  common::GHistIndexMatrix gmat;
  gmat.Init(qu, &ctx, p_fmat.get(), n_bins);

  RegTree tree;
  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  ExpandEntry node(ExpandEntry::kRootNid, tree.GetDepth(ExpandEntry::kRootNid));

  auto* row_set_collection = updater.TestInitData(gmat, gpair, *p_fmat, tree);
  auto& row_idxs = row_set_collection->Data();
  const size_t* row_idxs_ptr = row_idxs.DataConst();
  updater.TestBuildHistogramsLossGuide(node, gmat, &tree, gpair);
  const auto snode = updater.TestInitNewNode(ExpandEntry::kRootNid, gmat, gpair, tree);

  GradStats<GradientSumT> grad_stat;
  {
    ::sycl::buffer<GradStats<GradientSumT>> buff(&grad_stat, 1);
    qu->submit([&](::sycl::handler& cgh) {
      auto buff_acc  = buff.template get_access<::sycl::access::mode::read_write>(cgh);
      cgh.single_task<>([=]() {
        for (size_t i = 0; i < num_rows; ++i) {
          size_t row_idx = row_idxs_ptr[i];
          buff_acc[0] += GradStats<GradientSumT>(gpair_ptr[row_idx].GetGrad(),
                                                 gpair_ptr[row_idx].GetHess());
        }
      });
    }).wait_and_throw();
  }

  EXPECT_NEAR(snode.stats.GetGrad(), grad_stat.GetGrad(), 1e-6 * grad_stat.GetGrad());
  EXPECT_NEAR(snode.stats.GetHess(), grad_stat.GetHess(), 1e-6 * grad_stat.GetHess());
}

template <typename GradientSumT>
void TestHistUpdaterEvaluateSplits(const xgboost::tree::TrainParam& param) {
  const size_t num_rows = 1u << 8;
  const size_t num_columns = 2;
  const size_t n_bins = 32;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  auto p_fmat = RandomDataGenerator{num_rows, num_columns, 0.0f}.GenerateDMatrix();

  FeatureInteractionConstraintHost int_constraints;

  TestHistUpdater<GradientSumT> updater(&ctx, qu, param, int_constraints, p_fmat.get());
  updater.SetHistSynchronizer(new BatchHistSynchronizer<GradientSumT>());
  updater.SetHistRowsAdder(new BatchHistRowsAdder<GradientSumT>());

  HostDeviceVector<GradientPair> gpair(num_rows, {0, 0}, ctx.Device());
  auto* gpair_ptr = gpair.DevicePointer();
  GenerateRandomGPairs(qu, gpair_ptr, num_rows, false);

  common::GHistIndexMatrix gmat;
  gmat.Init(qu, &ctx, p_fmat.get(), n_bins);

  RegTree tree;
  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);
  ExpandEntry node(ExpandEntry::kRootNid, tree.GetDepth(ExpandEntry::kRootNid));

  auto* row_set_collection = updater.TestInitData(gmat, gpair, *p_fmat, tree);
  auto& row_idxs = row_set_collection->Data();
  const size_t* row_idxs_ptr = row_idxs.DataConst();
  const auto* hist = updater.TestBuildHistogramsLossGuide(node, gmat, &tree, gpair);
  const auto snode_init = updater.TestInitNewNode(ExpandEntry::kRootNid, gmat, gpair, tree);

  const auto snode_updated = updater.TestEvaluateSplits({node}, gmat, tree);
  auto best_loss_chg = snode_updated[0].best.loss_chg;
  auto stats = snode_init.stats;
  auto root_gain = snode_init.root_gain;

  // Check all splits manually. Save the best one and compare with the ans
  TreeEvaluator<GradientSumT> tree_evaluator(qu, param, num_columns);
  auto evaluator = tree_evaluator.GetEvaluator();
  const uint32_t* cut_ptr = gmat.cut.cut_ptrs_.ConstDevicePointer();
  const size_t size = gmat.cut.cut_ptrs_.Size();
  int n_better_splits = 0;
  const auto* hist_ptr = (*hist)[0].DataConst();
  std::vector<bst_float> best_loss_chg_des(1, -1);
  {
    ::sycl::buffer<bst_float> best_loss_chg_buff(best_loss_chg_des.data(), 1);
    qu->submit([&](::sycl::handler& cgh) {
      auto best_loss_chg_acc = best_loss_chg_buff.template get_access<::sycl::access::mode::read_write>(cgh);
      cgh.single_task<>([=]() {
        for (size_t i = 1; i < size; ++i) {
          GradStats<GradientSumT> left(0, 0);
          GradStats<GradientSumT> right = stats - left;
          for (size_t j = cut_ptr[i-1]; j < cut_ptr[i]; ++j) {
            auto loss_change = evaluator.CalcSplitGain(0, i - 1, left, right) - root_gain;
            if (loss_change > best_loss_chg_acc[0]) {
              best_loss_chg_acc[0] = loss_change;
            }
            left.Add(hist_ptr[j].GetGrad(), hist_ptr[j].GetHess());
            right = stats - left;
          }
        }
      });
    }).wait();
  }

  ASSERT_NEAR(best_loss_chg_des[0], best_loss_chg, 1e-4);
}

template <typename GradientSumT>
void TestHistUpdaterApplySplit(const xgboost::tree::TrainParam& param, float sparsity, int max_bins) {
  const size_t num_rows = 1024;
  const size_t num_columns = 2;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  auto p_fmat = RandomDataGenerator{num_rows, num_columns, sparsity}.GenerateDMatrix();
  common::GHistIndexMatrix gmat;
  gmat.Init(qu, &ctx, p_fmat.get(), max_bins);

  RegTree tree;
  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);

  std::vector<tree::ExpandEntry> nodes;
  nodes.emplace_back(tree::ExpandEntry(0, tree.GetDepth(0)));

  FeatureInteractionConstraintHost int_constraints;
  TestHistUpdater<GradientSumT> updater(&ctx, qu, param, int_constraints, p_fmat.get());
  HostDeviceVector<GradientPair> gpair(num_rows, {0, 0}, ctx.Device());
  auto* gpair_ptr = gpair.DevicePointer();
  GenerateRandomGPairs(qu, gpair_ptr, num_rows, false);

  auto* row_set_collection = updater.TestInitData(gmat, gpair, *p_fmat, tree);
  updater.TestApplySplit(nodes, gmat, &tree);

  // Copy indexes to host
  std::vector<size_t> row_indices_host(num_rows);
  qu->memcpy(row_indices_host.data(), row_set_collection->Data().Data(), sizeof(size_t)*num_rows).wait();

  // Reference Implementation
  std::vector<size_t> row_indices_desired_host(num_rows);
  size_t n_left, n_right;
  {
    TestHistUpdater<GradientSumT> updater4verification(&ctx, qu, param, int_constraints, p_fmat.get());
    auto* row_set_collection4verification = updater4verification.TestInitData(gmat, gpair, *p_fmat, tree);

    size_t n_nodes = nodes.size();
    std::vector<int32_t> split_conditions(n_nodes);
    xgboost::tree::CommonRowPartitioner::FindSplitConditions(nodes, tree, gmat, &split_conditions);

    common::PartitionBuilder partition_builder;
    partition_builder.Init(qu, n_nodes, [&](size_t node_in_set) {
      const int32_t nid = nodes[node_in_set].nid;
      return (*row_set_collection4verification)[nid].Size();
    });

    ::sycl::event event;
    partition_builder.Partition(gmat, nodes, (*row_set_collection4verification),
                                split_conditions, &tree, &event);
    qu->wait_and_throw();

    for (size_t node_in_set = 0; node_in_set < n_nodes; node_in_set++) {
      const int32_t nid = nodes[node_in_set].nid;
      size_t* data_result = const_cast<size_t*>((*row_set_collection4verification)[nid].begin);
      partition_builder.MergeToArray(node_in_set, data_result, &event);
    }
    qu->wait_and_throw();

    const int32_t nid = nodes[0].nid;
    n_left = partition_builder.GetNLeftElems(0);
    n_right = partition_builder.GetNRightElems(0);

    row_set_collection4verification->AddSplit(nid, tree[nid].LeftChild(),
        tree[nid].RightChild(), n_left, n_right);

    qu->memcpy(row_indices_desired_host.data(), row_set_collection4verification->Data().Data(), sizeof(size_t)*num_rows).wait();
  }

  std::sort(row_indices_desired_host.begin(), row_indices_desired_host.begin() + n_left);
  std::sort(row_indices_host.begin(), row_indices_host.begin() + n_left);
  std::sort(row_indices_desired_host.begin() + n_left, row_indices_desired_host.end());
  std::sort(row_indices_host.begin() + n_left, row_indices_host.end());

  for (size_t row = 0; row < num_rows; ++row) {
    ASSERT_EQ(row_indices_desired_host[row], row_indices_host[row]);
  }
}

template <typename GradientSumT>
void TestHistUpdaterExpandWithLossGuide(const xgboost::tree::TrainParam& param) {
  const size_t num_rows = 3;
  const size_t num_columns = 1;
  const size_t n_bins = 16;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  std::vector<float> data = {7, 3, 15};
  auto p_fmat = GetDMatrixFromData(data, num_rows, num_columns);
  common::GHistIndexMatrix gmat;
  gmat.Init(qu, &ctx, p_fmat.get(), n_bins);

  HostDeviceVector<GradientPair> gpair({{1, 2}, {3, 1}, {1, 1}}, ctx.Device());

  RegTree tree;
  FeatureInteractionConstraintHost int_constraints;
  TestHistUpdater<GradientSumT> updater(&ctx, qu, param, int_constraints, p_fmat.get());
  updater.SetHistSynchronizer(new BatchHistSynchronizer<GradientSumT>());
  updater.SetHistRowsAdder(new BatchHistRowsAdder<GradientSumT>());
  auto* row_set_collection = updater.TestInitData(gmat, gpair, *p_fmat, tree);

  updater.TestExpandWithLossGuide(gmat, p_fmat.get(), &tree, gpair);

  const auto& nodes = tree.GetNodes();
  std::vector<float> ans(data.size());
  for (size_t data_idx = 0; data_idx < data.size(); ++data_idx) {
      size_t node_idx = 0;
      while (!nodes[node_idx].IsLeaf()) {
        node_idx = data[data_idx] < nodes[node_idx].SplitCond() ? nodes[node_idx].LeftChild() : nodes[node_idx].RightChild();
      }
      ans[data_idx] = nodes[node_idx].LeafValue();
  }

  ASSERT_NEAR(ans[0], -0.15, 1e-6);
  ASSERT_NEAR(ans[1], -0.45, 1e-6);
  ASSERT_NEAR(ans[2], -0.15, 1e-6);
}


template <typename GradientSumT>
void TestHistUpdaterExpandWithDepthWise(const xgboost::tree::TrainParam& param) {
  const size_t num_rows = 3;
  const size_t num_columns = 1;
  const size_t n_bins = 16;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  std::vector<float> data = {7, 3, 15};
  auto p_fmat = GetDMatrixFromData(data, num_rows, num_columns);
  common::GHistIndexMatrix gmat;
  gmat.Init(qu, &ctx, p_fmat.get(), n_bins);

  HostDeviceVector<GradientPair> gpair({{1, 2}, {3, 1}, {1, 1}}, ctx.Device());

  RegTree tree;
  FeatureInteractionConstraintHost int_constraints;
  TestHistUpdater<GradientSumT> updater(&ctx, qu, param, int_constraints, p_fmat.get());
  updater.SetHistSynchronizer(new BatchHistSynchronizer<GradientSumT>());
  updater.SetHistRowsAdder(new BatchHistRowsAdder<GradientSumT>());
  auto* row_set_collection = updater.TestInitData(gmat, gpair, *p_fmat, tree);

  updater.TestExpandWithDepthWise(gmat, p_fmat.get(), &tree, gpair);

  const auto& nodes = tree.GetNodes();
  std::vector<float> ans(data.size());
  for (size_t data_idx = 0; data_idx < data.size(); ++data_idx) {
      size_t node_idx = 0;
      while (!nodes[node_idx].IsLeaf()) {
        node_idx = data[data_idx] < nodes[node_idx].SplitCond() ? nodes[node_idx].LeftChild() : nodes[node_idx].RightChild();
      }
      ans[data_idx] = nodes[node_idx].LeafValue();
  }

  ASSERT_NEAR(ans[0], -0.15, 1e-6);
  ASSERT_NEAR(ans[1], -0.45, 1e-6);
  ASSERT_NEAR(ans[2], -0.15, 1e-6);
}

TEST(SyclHistUpdater, Sampling) {
  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"subsample", "0.7"}});

  TestHistUpdaterSampling<float>(param);
  TestHistUpdaterSampling<double>(param);
}

TEST(SyclHistUpdater, InitData) {
  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"subsample", "1"}});

  TestHistUpdaterInitData<float>(param, true);
  TestHistUpdaterInitData<float>(param, false);

  TestHistUpdaterInitData<double>(param, true);
  TestHistUpdaterInitData<double>(param, false);
}

TEST(SyclHistUpdater, BuildHistogramsLossGuide) {
  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_depth", "3"}});

  TestHistUpdaterBuildHistogramsLossGuide<float>(param, 0.0);
  TestHistUpdaterBuildHistogramsLossGuide<float>(param, 0.5);
  TestHistUpdaterBuildHistogramsLossGuide<double>(param, 0.0);
  TestHistUpdaterBuildHistogramsLossGuide<double>(param, 0.5);
}

TEST(SyclHistUpdater, InitNewNode) {
  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_depth", "3"}});

  TestHistUpdaterInitNewNode<float>(param, 0.0);
  TestHistUpdaterInitNewNode<float>(param, 0.5);
  TestHistUpdaterInitNewNode<double>(param, 0.0);
  TestHistUpdaterInitNewNode<double>(param, 0.5);
}

TEST(SyclHistUpdater, EvaluateSplits) {
  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_depth", "3"}});

  TestHistUpdaterEvaluateSplits<float>(param);
  TestHistUpdaterEvaluateSplits<double>(param);
}

TEST(SyclHistUpdater, ApplySplitSparce) {
  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_depth", "3"}});

  TestHistUpdaterApplySplit<float>(param, 0.3, 256);
  TestHistUpdaterApplySplit<double>(param, 0.3, 256);
}

TEST(SyclHistUpdater, ApplySplitDence) {
  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_depth", "3"}});

  TestHistUpdaterApplySplit<float>(param, 0.0, 256);
  TestHistUpdaterApplySplit<float>(param, 0.0, 256+1);
  TestHistUpdaterApplySplit<float>(param, 0.0, (1u << 16) + 1);
  TestHistUpdaterApplySplit<double>(param, 0.0, 256);
  TestHistUpdaterApplySplit<double>(param, 0.0, 256+1);
  TestHistUpdaterApplySplit<double>(param, 0.0, (1u << 16) + 1);
}

TEST(SyclHistUpdater, ExpandWithLossGuide) {
  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_depth", "2"},
                                {"grow_policy", "lossguide"}});

  TestHistUpdaterExpandWithLossGuide<float>(param);
  TestHistUpdaterExpandWithLossGuide<double>(param);
}

TEST(SyclHistUpdater, ExpandWithDepthWise) {
  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_depth", "2"}});

  TestHistUpdaterExpandWithDepthWise<float>(param);
  TestHistUpdaterExpandWithDepthWise<double>(param);
}

}  // namespace xgboost::sycl::tree
