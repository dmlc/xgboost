#include "test_ranking_obj.cc"

#include "../../../src/objective/rank_obj.cu"

namespace xgboost {

template <typename T = uint32_t, typename Comparator = thrust::greater<T>>
std::unique_ptr<dh::SegmentSorter<T>>
RankSegmentSorterTestImpl(const std::vector<uint32_t> &group_indices,
                          const std::vector<T> &hlabels,
                          const std::vector<T> &expected_sorted_hlabels,
                          const std::vector<uint32_t> &expected_orig_pos
                          ) {
  std::unique_ptr<dh::SegmentSorter<T>> seg_sorter_ptr(new dh::SegmentSorter<T>);
  dh::SegmentSorter<T> &seg_sorter(*seg_sorter_ptr);

  // Create a bunch of unsorted labels on the device and sort it via the segment sorter
  dh::device_vector<T> dlabels(hlabels);
  seg_sorter.SortItems(dlabels.data().get(), dlabels.size(), group_indices, Comparator());

  auto num_items = seg_sorter.GetItemsSpan().size();
  EXPECT_EQ(num_items, group_indices.back());
  EXPECT_EQ(seg_sorter.GetNumGroups(), group_indices.size() - 1);

  // Check the labels
  dh::device_vector<T> sorted_dlabels(num_items);
  sorted_dlabels.assign(dh::tcbegin(seg_sorter.GetItemsSpan()),
                        dh::tcend(seg_sorter.GetItemsSpan()));
  thrust::host_vector<T> sorted_hlabels(sorted_dlabels);
  EXPECT_EQ(expected_sorted_hlabels, sorted_hlabels);

  // Check the indices
  dh::device_vector<uint32_t> dorig_pos(num_items);
  dorig_pos.assign(dh::tcbegin(seg_sorter.GetOriginalPositionsSpan()),
                   dh::tcend(seg_sorter.GetOriginalPositionsSpan()));
  dh::device_vector<uint32_t> horig_pos(dorig_pos);
  EXPECT_EQ(expected_orig_pos, horig_pos);

  return seg_sorter_ptr;
}

TEST(Objective, RankSegmentSorterTest) {
  RankSegmentSorterTestImpl({0, 2, 4, 7, 10, 14, 18, 22, 26},  // Groups
                            {1, 1,                             // Labels
                             1, 2,
                             3, 2, 1,
                             1, 2, 1,
                             1, 3, 4, 2,
                             1, 2, 1, 1,
                             1, 2, 2, 3,
                             3, 3, 1, 2},
                            {1, 1,                             // Expected sorted labels
                             2, 1,
                             3, 2, 1,
                             2, 1, 1,
                             4, 3, 2, 1,
                             2, 1, 1, 1,
                             3, 2, 2, 1,
                             3, 3, 2, 1},
                            {0, 1,                             // Expected original positions
                             3, 2,
                             4, 5, 6,
                             8, 7, 9,
                             12, 11, 13, 10,
                             15, 14, 16, 17,
                             21, 19, 20, 18,
                             22, 23, 25, 24});
}

TEST(Objective, RankSegmentSorterSingleGroupTest) {
  RankSegmentSorterTestImpl({0, 7},                  // Groups
                            {6, 1, 4, 3, 0, 5, 2},   // Labels
                            {6, 5, 4, 3, 2, 1, 0},   // Expected sorted labels
                            {0, 5, 2, 3, 6, 1, 4});  // Expected original positions
}

TEST(Objective, RankSegmentSorterAscendingTest) {
  RankSegmentSorterTestImpl<uint32_t, thrust::less<uint32_t>>(
                                                    {0, 4, 7},    // Groups
                                                    {3, 1, 4, 2,  // Labels
                                                     6, 5, 7},
                                                    {1, 2, 3, 4,  // Expected sorted labels
                                                     5, 6, 7},
                                                    {1, 3, 0, 2,  // Expected original positions
                                                     5, 4, 6});
}

using CountFunctor = uint32_t (*)(const int *, uint32_t, int);
void RankItemCountImpl(const std::vector<int> &sorted_items, CountFunctor f,
                       int find_val, uint32_t exp_val) {
  EXPECT_NE(std::find(sorted_items.begin(), sorted_items.end(), find_val), sorted_items.end());
  EXPECT_EQ(f(&sorted_items[0], sorted_items.size(), find_val), exp_val);
}

TEST(Objective, RankItemCountOnLeft) {
  // Items sorted descendingly
  std::vector<int> sorted_items{10, 10, 6, 4, 4, 4, 4, 1, 1, 1, 1, 1, 0};
  RankItemCountImpl(sorted_items, &xgboost::obj::CountNumItemsToTheLeftOf,
                    10, static_cast<uint32_t>(0));
  RankItemCountImpl(sorted_items, &xgboost::obj::CountNumItemsToTheLeftOf,
                    6, static_cast<uint32_t>(2));
  RankItemCountImpl(sorted_items, &xgboost::obj::CountNumItemsToTheLeftOf,
                    4, static_cast<uint32_t>(3));
  RankItemCountImpl(sorted_items, &xgboost::obj::CountNumItemsToTheLeftOf,
                    1, static_cast<uint32_t>(7));
  RankItemCountImpl(sorted_items, &xgboost::obj::CountNumItemsToTheLeftOf,
                    0, static_cast<uint32_t>(12));
}

TEST(Objective, RankItemCountOnRight) {
  // Items sorted descendingly
  std::vector<int> sorted_items{10, 10, 6, 4, 4, 4, 4, 1, 1, 1, 1, 1, 0};
  RankItemCountImpl(sorted_items, &xgboost::obj::CountNumItemsToTheRightOf,
                    10, static_cast<uint32_t>(11));
  RankItemCountImpl(sorted_items, &xgboost::obj::CountNumItemsToTheRightOf,
                    6, static_cast<uint32_t>(10));
  RankItemCountImpl(sorted_items, &xgboost::obj::CountNumItemsToTheRightOf,
                    4, static_cast<uint32_t>(6));
  RankItemCountImpl(sorted_items, &xgboost::obj::CountNumItemsToTheRightOf,
                    1, static_cast<uint32_t>(1));
  RankItemCountImpl(sorted_items, &xgboost::obj::CountNumItemsToTheRightOf,
                    0, static_cast<uint32_t>(0));
}

TEST(Objective, NDCGLambdaWeightComputerTest) {
  std::vector<float> hlabels = {3.1f, 1.2f, 2.3f, 4.4f,        // Labels
                                7.8f, 5.01f, 6.96f,
                                10.3f, 8.7f, 11.4f, 9.45f, 11.4f};
  dh::device_vector<bst_float> dlabels(hlabels);

  auto segment_label_sorter = RankSegmentSorterTestImpl<float>(
    {0, 4, 7, 12},                  // Groups
    hlabels,
    {4.4f, 3.1f, 2.3f, 1.2f,        // Expected sorted labels
     7.8f, 6.96f, 5.01f,
     11.4f, 11.4f, 10.3f, 9.45f, 8.7f},
    {3, 0, 2, 1,                    // Expected original positions
     4, 6, 5,
     9, 11, 7, 10, 8});

  // Created segmented predictions for the labels from above
  std::vector<bst_float> hpreds{-9.78f, 24.367f, 0.908f, -11.47f,
                                -1.03f, -2.79f, -3.1f,
                                104.22f, 103.1f, -101.7f, 100.5f, 45.1f};
  dh::device_vector<bst_float> dpreds(hpreds);

  xgboost::obj::NDCGLambdaWeightComputer ndcg_lw_computer(dpreds.data().get(),
                                                          dlabels.data().get(),
                                                          *segment_label_sorter);

  // Where will the predictions move from its current position, if they were sorted
  // descendingly?
  auto dsorted_pred_pos = ndcg_lw_computer.GetPredictionSorter().GetIndexableSortedPositionsSpan();
  std::vector<uint32_t> hsorted_pred_pos(segment_label_sorter->GetNumItems());
  dh::CopyDeviceSpanToVector(&hsorted_pred_pos, dsorted_pred_pos);
  std::vector<uint32_t> expected_sorted_pred_pos{2, 0, 1, 3,
                                                 4, 5, 6,
                                                 7, 8, 11, 9, 10};
  EXPECT_EQ(expected_sorted_pred_pos, hsorted_pred_pos);

  // Check group DCG values
  std::vector<float> hgroup_dcgs(segment_label_sorter->GetNumGroups());
  dh::CopyDeviceSpanToVector(&hgroup_dcgs, ndcg_lw_computer.GetGroupDcgsSpan());
  std::vector<uint32_t> hgroups(segment_label_sorter->GetNumGroups() + 1);
  dh::CopyDeviceSpanToVector(&hgroups, segment_label_sorter->GetGroupsSpan());
  EXPECT_EQ(hgroup_dcgs.size(), segment_label_sorter->GetNumGroups());
  std::vector<float> hsorted_labels(segment_label_sorter->GetNumItems());
  dh::CopyDeviceSpanToVector(&hsorted_labels, segment_label_sorter->GetItemsSpan());
  for (auto i = 0; i < hgroup_dcgs.size(); ++i) {
    // Compute group DCG value on CPU and compare
    auto gbegin = hgroups[i];
    auto gend = hgroups[i + 1];
    EXPECT_NEAR(
      hgroup_dcgs[i],
      xgboost::obj::NDCGLambdaWeightComputer::ComputeGroupDCGWeight(&hsorted_labels[gbegin],
                                                                    gend - gbegin),
      0.01f);
  }
}

TEST(Objective, IndexableSortedItemsTest) {
  std::vector<float> hlabels = {3.1f, 1.2f, 2.3f, 4.4f,        // Labels
                                7.8f, 5.01f, 6.96f,
                                10.3f, 8.7f, 11.4f, 9.45f, 11.4f};
  dh::device_vector<bst_float> dlabels(hlabels);

  auto segment_label_sorter = RankSegmentSorterTestImpl<float>(
    {0, 4, 7, 12},                  // Groups
    hlabels,
    {4.4f, 3.1f, 2.3f, 1.2f,        // Expected sorted labels
     7.8f, 6.96f, 5.01f,
     11.4f, 11.4f, 10.3f, 9.45f, 8.7f},
    {3, 0, 2, 1,                    // Expected original positions
     4, 6, 5,
     9, 11, 7, 10, 8});

  segment_label_sorter->CreateIndexableSortedPositions();
  std::vector<uint32_t> sorted_indices(segment_label_sorter->GetNumItems());
  dh::CopyDeviceSpanToVector(&sorted_indices,
                             segment_label_sorter->GetIndexableSortedPositionsSpan());
  std::vector<uint32_t> expected_sorted_indices = {
    1, 3, 2, 0,
    4, 6, 5,
    9, 11, 7, 10, 8};
  EXPECT_EQ(expected_sorted_indices, sorted_indices);
}

TEST(Objective, ComputeAndCompareMAPStatsTest) {
  std::vector<float> hlabels = {3.1f, 0.0f, 2.3f, 4.4f,        // Labels
                                0.0f, 5.01f, 0.0f,
                                10.3f, 0.0f, 11.4f, 9.45f, 11.4f};
  dh::device_vector<bst_float> dlabels(hlabels);

  auto segment_label_sorter = RankSegmentSorterTestImpl<float>(
    {0, 4, 7, 12},                  // Groups
    hlabels,
    {4.4f, 3.1f, 2.3f, 0.0f,        // Expected sorted labels
     5.01f, 0.0f, 0.0f,
     11.4f, 11.4f, 10.3f, 9.45f, 0.0f},
    {3, 0, 2, 1,                    // Expected original positions
     5, 4, 6,
     9, 11, 7, 10, 8});

  // Create MAP stats on the device first using the objective
  std::vector<bst_float> hpreds{-9.78f, 24.367f, 0.908f, -11.47f,
                                -1.03f, -2.79f, -3.1f,
                                104.22f, 103.1f, -101.7f, 100.5f, 45.1f};
  dh::device_vector<bst_float> dpreds(hpreds);

  xgboost::obj::MAPLambdaWeightComputer map_lw_computer(dpreds.data().get(),
                                                        dlabels.data().get(),
                                                        *segment_label_sorter);

  // Get the device MAP stats on host
  std::vector<xgboost::obj::MAPLambdaWeightComputer::MAPStats> dmap_stats(
    segment_label_sorter->GetNumItems());
  dh::CopyDeviceSpanToVector(&dmap_stats, map_lw_computer.GetMapStatsSpan());

  // Compute the MAP stats on host next to compare
  std::vector<uint32_t> hgroups(segment_label_sorter->GetNumGroups() + 1);
  dh::CopyDeviceSpanToVector(&hgroups, segment_label_sorter->GetGroupsSpan());

  for (auto i = 0; i < hgroups.size() - 1; ++i) {
    auto gbegin = hgroups[i];
    auto gend = hgroups[i + 1];
    std::vector<xgboost::obj::ListEntry> lst_entry;
    for (auto j = gbegin; j < gend; ++j) {
      lst_entry.emplace_back(hpreds[j], hlabels[j], j);
    }
    std::stable_sort(lst_entry.begin(), lst_entry.end(), xgboost::obj::ListEntry::CmpPred);

    // Compute the MAP stats with this list and compare with the ones computed on the device
    std::vector<xgboost::obj::MAPLambdaWeightComputer::MAPStats> hmap_stats;
    xgboost::obj::MAPLambdaWeightComputer::GetMAPStats(lst_entry, &hmap_stats);
    for (auto j = gbegin; j < gend; ++j) {
      EXPECT_EQ(dmap_stats[j].hits, hmap_stats[j - gbegin].hits);
      EXPECT_NEAR(dmap_stats[j].ap_acc, hmap_stats[j - gbegin].ap_acc, 0.01f);
      EXPECT_NEAR(dmap_stats[j].ap_acc_miss, hmap_stats[j - gbegin].ap_acc_miss, 0.01f);
      EXPECT_NEAR(dmap_stats[j].ap_acc_add, hmap_stats[j - gbegin].ap_acc_add, 0.01f);
    }
  }
}

}  // namespace xgboost
