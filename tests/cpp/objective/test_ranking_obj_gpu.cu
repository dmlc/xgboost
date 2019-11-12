#include "test_ranking_obj.cc"

#include "../../../src/objective/rank_obj.cu"

namespace xgboost {

template <typename T = uint32_t, typename Comparator = thrust::greater<T>>
std::unique_ptr<xgboost::obj::SegmentSorter<T>>
RankSegmentSorterTestImpl(const std::vector<uint32_t> &group_indices,
                          const std::vector<T> &hlabels,
                          const std::vector<T> &expected_sorted_hlabels,
                          const std::vector<uint32_t> &expected_orig_pos
                          ) {
  std::unique_ptr<xgboost::obj::SegmentSorter<T>> seg_sorter_ptr(
    new xgboost::obj::SegmentSorter<T>);
  xgboost::obj::SegmentSorter<T> &seg_sorter(*seg_sorter_ptr);

  // Create a bunch of unsorted labels on the device and sort it via the segment sorter
  dh::device_vector<T> dlabels(hlabels);
  seg_sorter.SortItems(dlabels.data().get(), dlabels.size(), group_indices, Comparator());

  EXPECT_EQ(seg_sorter.NumItems(), group_indices.back());
  EXPECT_EQ(seg_sorter.NumGroups(), group_indices.size() - 1);

  // Check the labels
  dh::device_vector<T> sorted_dlabels(seg_sorter.NumItems());
  sorted_dlabels.assign(thrust::device_ptr<const T>(seg_sorter.Items()),
                        thrust::device_ptr<const T>(seg_sorter.Items())
                        + seg_sorter.NumItems());
  thrust::host_vector<T> sorted_hlabels(sorted_dlabels);
  EXPECT_EQ(expected_sorted_hlabels, sorted_hlabels);

  // Check the indices
  dh::device_vector<uint32_t> dorig_pos(seg_sorter.NumItems());
  dorig_pos.assign(thrust::device_ptr<const uint32_t>(seg_sorter.OriginalPositions()),
                   thrust::device_ptr<const uint32_t>(seg_sorter.OriginalPositions())
                   + seg_sorter.NumItems());
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
  auto segment_label_sorter = RankSegmentSorterTestImpl<float>(
    {0, 4, 7, 12},                  // Groups
    {3.1f, 1.2f, 2.3f, 4.4f,        // Labels
     7.8f, 5.01f, 6.96f,
     10.3f, 8.7f, 11.4f, 9.45f, 11.4f},
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
                                                          dpreds.size(),
                                                          *segment_label_sorter);

  // Where will the predictions move from its current position, if they were sorted
  // descendingly?
  auto dsorted_pred_pos = ndcg_lw_computer.GetSortedPredPos();
  thrust::host_vector<uint32_t> hsorted_pred_pos(dsorted_pred_pos);
  std::vector<uint32_t> expected_sorted_pred_pos{2, 0, 1, 3,
                                                 4, 5, 6,
                                                 7, 8, 11, 9, 10};
  EXPECT_EQ(expected_sorted_pred_pos, hsorted_pred_pos);
}

}  // namespace xgboost
