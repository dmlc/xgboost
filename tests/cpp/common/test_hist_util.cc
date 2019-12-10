#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <utility>

#include "../../../src/common/hist_util.h"
#include "../helpers.h"

namespace xgboost {
namespace common {

TEST(CutsBuilder, SearchGroupInd) {
  size_t constexpr kNumGroups = 4;
  size_t constexpr kRows = 17;
  size_t constexpr kCols = 15;

  auto pp_dmat = CreateDMatrix(kRows, kCols, 0);
  std::shared_ptr<DMatrix> p_mat {*pp_dmat};

  std::vector<bst_int> group(kNumGroups);
  group[0] = 2;
  group[1] = 3;
  group[2] = 7;
  group[3] = 5;

  p_mat->Info().SetInfo(
      "group", group.data(), DataType::kUInt32, kNumGroups);

  HistogramCuts hmat;

  size_t group_ind = CutsBuilder::SearchGroupIndFromRow(p_mat->Info().group_ptr_, 0);
  ASSERT_EQ(group_ind, 0);

  group_ind = CutsBuilder::SearchGroupIndFromRow(p_mat->Info().group_ptr_, 5);
  ASSERT_EQ(group_ind, 2);

  EXPECT_ANY_THROW(CutsBuilder::SearchGroupIndFromRow(p_mat->Info().group_ptr_, 17));

  delete pp_dmat;
}

namespace {
class SparseCutsWrapper : public SparseCuts {
 public:
  std::vector<uint32_t> const& ColPtrs()   const { return p_cuts_->Ptrs(); }
  std::vector<float>    const& ColValues() const { return p_cuts_->Values(); }
};
}  // anonymous namespace

TEST(SparseCuts, SingleThreadedBuild) {
  size_t constexpr kRows = 267;
  size_t constexpr kCols = 31;
  size_t constexpr kBins = 256;

  auto pp_dmat = CreateDMatrix(kRows, kCols, 0);
  std::shared_ptr<DMatrix> p_fmat {*pp_dmat};

  common::GHistIndexMatrix hmat;
  hmat.Init(p_fmat.get(), kBins);

  HistogramCuts cuts;
  SparseCuts indices(&cuts);
  auto const& page = *(p_fmat->GetBatches<xgboost::CSCPage>().begin());
  indices.SingleThreadBuild(page, p_fmat->Info(), kBins, false, 0, page.Size(), 0);

  ASSERT_EQ(hmat.cut.Ptrs().size(), cuts.Ptrs().size());
  ASSERT_EQ(hmat.cut.Ptrs(), cuts.Ptrs());
  ASSERT_EQ(hmat.cut.Values(), cuts.Values());
  ASSERT_EQ(hmat.cut.MinValues(), cuts.MinValues());

  delete pp_dmat;
}

TEST(SparseCuts, MultiThreadedBuild) {
  size_t constexpr kRows = 17;
  size_t constexpr kCols = 15;
  size_t constexpr kBins = 255;

  omp_ulong ori_nthreads = omp_get_max_threads();
  omp_set_num_threads(16);

  auto Compare =
#if defined(_MSC_VER)  // msvc fails to capture
      [kBins](DMatrix* p_fmat) {
#else
      [](DMatrix* p_fmat) {
#endif
        HistogramCuts threaded_container;
        SparseCuts threaded_indices(&threaded_container);
        threaded_indices.Build(p_fmat, kBins);

        HistogramCuts container;
        SparseCuts indices(&container);
        auto const& page = *(p_fmat->GetBatches<xgboost::CSCPage>().begin());
        indices.SingleThreadBuild(page, p_fmat->Info(), kBins, false, 0, page.Size(), 0);

        ASSERT_EQ(container.Ptrs().size(), threaded_container.Ptrs().size());
        ASSERT_EQ(container.Values().size(), threaded_container.Values().size());

        for (uint32_t i = 0; i < container.Ptrs().size(); ++i) {
          ASSERT_EQ(container.Ptrs()[i], threaded_container.Ptrs()[i]);
        }
        for (uint32_t i = 0; i < container.Values().size(); ++i) {
          ASSERT_EQ(container.Values()[i], threaded_container.Values()[i]);
        }
      };

  {
    auto pp_mat = CreateDMatrix(kRows, kCols, 0);
    DMatrix* p_fmat = (*pp_mat).get();
    Compare(p_fmat);
    delete pp_mat;
  }

  {
    auto pp_mat = CreateDMatrix(kRows, kCols, 0.0001);
    DMatrix* p_fmat = (*pp_mat).get();
    Compare(p_fmat);
    delete pp_mat;
  }

  omp_set_num_threads(ori_nthreads);
}

}  // namespace common
}  // namespace xgboost
