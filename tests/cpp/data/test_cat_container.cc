/**
 * Copyright 2025, XGBoost contributors
 */

#include "test_cat_container.h"

#include <dmlc/logging.h>  // for dmlc::Error
#include <gtest/gtest.h>

#include <array>    // for array
#include <cstdint>  // for int32_t, uint64_t
#include <vector>   // for vector

#include "../../../src/collective/allreduce.h"  // for Allreduce
#include "../../../src/data/cat_container.h"
#include "../../../src/data/cat_container_hash.h"  // for HashCatHostContent
#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "../encoder/df_mock.h"
#include "xgboost/collective/result.h"  // for SafeColl
#include "xgboost/json.h"               // for Json
#include "xgboost/linalg.h"             // for MakeVec

namespace xgboost {
using DfTest = enc::cpu_impl::DfTest;

auto eq_check = [](common::Span<bst_cat_t const> sorted_idx, std::vector<bst_cat_t> const& sol) {
  ASSERT_EQ(sorted_idx, common::Span{sol});
};

TEST(CatContainer, Str) {
  Context ctx;
  TestCatContainerStr<DfTest>(&ctx, eq_check);
}

TEST(CatContainer, Mixed) {
  Context ctx;
  TestCatContainerMixed<DfTest>(&ctx, eq_check);
}

namespace {
struct StrColumnForTest {
  std::vector<std::int32_t> offsets;
  std::vector<enc::CatCharT> values;
  std::vector<enc::HostCatIndexView> columns;
  std::vector<std::int32_t> feature_segments;

  StrColumnForTest(std::vector<std::int32_t> o, std::vector<enc::CatCharT> v,
                   std::vector<std::int32_t> seg)
      : offsets{std::move(o)}, values{std::move(v)}, feature_segments{std::move(seg)} {}

  enc::HostColumnsView AsView() {
    columns.clear();
    columns.emplace_back(enc::CatStrArrayView{common::Span{offsets}, common::Span{values}});
    return enc::HostColumnsView{common::Span{columns}, common::Span{feature_segments},
                                feature_segments.empty() ? 0 : feature_segments.back()};
  }
};
}  // namespace

TEST(CatContainer, ValidatesGoodStrOffsets) {
  // two strings: "ab" (offset 0..2) and "cde" (offset 2..5); offsets.back() == values.size()
  StrColumnForTest fx{std::vector<std::int32_t>{0, 2, 5},
                      std::vector<enc::CatCharT>{'a', 'b', 'c', 'd', 'e'},
                      std::vector<std::int32_t>{0, 2}};
  auto view = fx.AsView();
  auto build = [&] { CatContainer cats{view, /*is_ref=*/false}; };
  EXPECT_NO_THROW(build());
}

TEST(CatContainer, RejectsOverrunStrOffsets) {
  // last offset 6 exceeds values.size()=5; tail-overrun must be rejected
  StrColumnForTest fx{std::vector<std::int32_t>{0, 2, 6},
                      std::vector<enc::CatCharT>{'a', 'b', 'c', 'd', 'e'},
                      std::vector<std::int32_t>{0, 2}};
  auto view = fx.AsView();
  auto build = [&] { CatContainer cats{view, /*is_ref=*/false}; };
  EXPECT_THROW(build(), dmlc::Error);
}

TEST(CatContainer, RejectsNonZeroFirstOffset) {
  // arrow invariant offsets[0]==0; non-zero first offset corrupts SortNames substring
  StrColumnForTest fx{std::vector<std::int32_t>{1, 2, 5},
                      std::vector<enc::CatCharT>{'a', 'b', 'c', 'd', 'e'},
                      std::vector<std::int32_t>{0, 2}};
  auto view = fx.AsView();
  auto build = [&] { CatContainer cats{view, /*is_ref=*/false}; };
  EXPECT_THROW(build(), dmlc::Error);
}

TEST(CatContainer, RejectsNegativeOffset) {
  // negative int32 offset casts to huge size_t; per-offset non-negativity catches it
  StrColumnForTest fx{std::vector<std::int32_t>{0, -1, 5},
                      std::vector<enc::CatCharT>{'a', 'b', 'c', 'd', 'e'},
                      std::vector<std::int32_t>{0, 2}};
  auto view = fx.AsView();
  auto build = [&] { CatContainer cats{view, /*is_ref=*/false}; };
  EXPECT_THROW(build(), dmlc::Error);
}

TEST(CatContainer, RejectsNonMonotonicOffset) {
  // non-monotonic offsets[1]=3 > offsets[2]=2; tail-bound passes but substring
  // (r_beg=3, r_end=2) yields an OOB unsigned length in SortNames
  StrColumnForTest fx{std::vector<std::int32_t>{0, 3, 2, 5},
                      std::vector<enc::CatCharT>{'a', 'b', 'c', 'd', 'e'},
                      std::vector<std::int32_t>{0, 3}};
  auto view = fx.AsView();
  auto build = [&] { CatContainer cats{view, /*is_ref=*/false}; };
  EXPECT_THROW(build(), dmlc::Error);
}

TEST(CatContainer, SelfCopyReturnsAndPreservesContent) {
  // a.Copy(&ctx, a) must early-return before scoped_lock{a, a, a}; re-locking a
  // non-recursive mutex is UB so the failure mode is unspecified; assert content
  // preservation as the observable witness
  StrColumnForTest fx{std::vector<std::int32_t>{0, 2, 5},
                      std::vector<enc::CatCharT>{'a', 'b', 'c', 'd', 'e'},
                      std::vector<std::int32_t>{0, 2}};
  auto view = fx.AsView();
  CatContainer cats{view, /*is_ref=*/false};
  Context ctx;
  cats.Copy(&ctx, cats);
  ASSERT_EQ(cats.NumCatsTotal(), 2u);
}

// SortIsIdempotent and SelfCopyReturnsAndPreservesContent are CPU-only by design:
// both exercise the shared sort_mu_ + sorted_ idempotency logic in CatContainer; the
// CUDA-build sibling code path goes through the same private members (verified by
// the existing CatContainer.ThreadSafety GPU test which exercises Sort()).
TEST(CatContainer, SortIsIdempotent) {
  // calling Sort twice on the same container must yield the same sorted_idx_ on the
  // second call; short-circuit via the sorted_ flag guarded by sort_mu_
  StrColumnForTest fx{std::vector<std::int32_t>{0, 1, 3, 5},
                      std::vector<enc::CatCharT>{'c', 'a', 'b', 'd', 'e'},
                      std::vector<std::int32_t>{0, 3}};
  auto view = fx.AsView();
  CatContainer cats{view, /*is_ref=*/false};
  Context ctx;
  cats.Sort(&ctx);
  std::vector<bst_cat_t> idx_first;
  {
    auto span = cats.RefSortedIndex(&ctx);
    idx_first.assign(span.cbegin(), span.cend());
  }
  cats.Sort(&ctx);
  std::vector<bst_cat_t> idx_second;
  {
    auto span = cats.RefSortedIndex(&ctx);
    idx_second.assign(span.cbegin(), span.cend());
  }
  ASSERT_EQ(idx_first, idx_second);
}

TEST(CatContainer, SaveLoadIsRefAndSortedKeysPersist) {
  // Save writes is_ref+sorted as JSON booleans; Load reads them back. Direct JSON
  // inspection on Save proves the keys land; re-Save on the loaded container with
  // identical-key inspection proves Load read both flags into the destination.
  StrColumnForTest fx{std::vector<std::int32_t>{0, 1, 3, 5},
                      std::vector<enc::CatCharT>{'c', 'a', 'b', 'd', 'e'},
                      std::vector<std::int32_t>{0, 3}};
  auto view = fx.AsView();
  CatContainer src{view, /*is_ref=*/true};
  Context ctx;
  src.Sort(&ctx);
  Json saved{Object{}};
  src.Save(&saved);

  // Save side: both flags appear in JSON with the right value
  auto const& obj = get<Object const>(saved);
  ASSERT_TRUE(obj.find("is_ref") != obj.cend());
  ASSERT_TRUE(obj.find("sorted") != obj.cend());
  ASSERT_TRUE(get<Boolean const>(obj.at("is_ref")));
  ASSERT_TRUE(get<Boolean const>(obj.at("sorted")));

  // Load side: is_ref restored (NeedRecode reflects the flag); re-Save round-trip
  // on dst proves Load also wrote sorted_ to the destination, since dst.Save()
  // serialises the in-memory sorted_ field
  CatContainer dst;
  dst.Load(saved);
  ASSERT_EQ(dst.NumCatsTotal(), src.NumCatsTotal());
  ASSERT_FALSE(dst.NeedRecode());

  Json resaved{Object{}};
  dst.Save(&resaved);
  auto const& robj = get<Object const>(resaved);
  ASSERT_TRUE(get<Boolean const>(robj.at("is_ref")));
  ASSERT_TRUE(get<Boolean const>(robj.at("sorted")));
}

TEST(CatContainer, LoadBackCompatIsRefDefaultsFalse) {
  // pre-PR JSON lacks is_ref; Load must default to false so NeedRecode() fires at
  // predict (the safe value -- a stale is_ref=true would skip required recoding)
  StrColumnForTest fx{std::vector<std::int32_t>{0, 1, 3, 5},
                      std::vector<enc::CatCharT>{'c', 'a', 'b', 'd', 'e'},
                      std::vector<std::int32_t>{0, 3}};
  auto view = fx.AsView();
  CatContainer src{view, /*is_ref=*/true};
  Json saved{Object{}};
  src.Save(&saved);
  // simulate pre-PR JSON by removing the new keys
  auto& obj = get<Object>(saved);
  obj.erase("is_ref");
  obj.erase("sorted");

  CatContainer dst;
  dst.Load(saved);
  ASSERT_TRUE(dst.NeedRecode());
}

TEST(CatContainerHash, DistinguishesIdenticalVsDivergentContent) {
  // identical inputs hash equal; divergent inputs hash differ (the distributed
  // consistency check in QuantileDMatrix construction relies on this contract)
  StrColumnForTest fx_a{std::vector<std::int32_t>{0, 2, 5},
                        std::vector<enc::CatCharT>{'a', 'b', 'c', 'd', 'e'},
                        std::vector<std::int32_t>{0, 2}};
  StrColumnForTest fx_a_copy{std::vector<std::int32_t>{0, 2, 5},
                             std::vector<enc::CatCharT>{'a', 'b', 'c', 'd', 'e'},
                             std::vector<std::int32_t>{0, 2}};
  StrColumnForTest fx_b{std::vector<std::int32_t>{0, 2, 5},
                        std::vector<enc::CatCharT>{'x', 'b', 'c', 'd', 'e'},
                        std::vector<std::int32_t>{0, 2}};

  auto hash_a = data::HashCatHostContent(fx_a.AsView());
  auto hash_a_copy = data::HashCatHostContent(fx_a_copy.AsView());
  auto hash_b = data::HashCatHostContent(fx_b.AsView());
  ASSERT_EQ(hash_a, hash_a_copy);
  ASSERT_NE(hash_a, hash_b);
}

TEST(CatContainerHash, DistributedDivergentDictsAllreduceDiffer) {
  // two workers with byte-divergent dictionaries; after kMin+kMax Allreduce on the
  // dual-component digest, lo != hi on at least one component, firing the production
  // CHECK_EQ in quantile_dmatrix.{cc,cu}
  constexpr std::int32_t kWorkers = 2;
  collective::TestDistributedGlobal(kWorkers, [] {
    auto rank = collective::GetRank();
    // rank 0 sees dictionary "ab"/"cde"; rank 1 sees "xy"/"cde" (first 2 bytes differ)
    std::vector<enc::CatCharT> values_r0{'a', 'b', 'c', 'd', 'e'};
    std::vector<enc::CatCharT> values_r1{'x', 'y', 'c', 'd', 'e'};
    std::vector<std::int32_t> offsets{0, 2, 5};
    std::vector<std::int32_t> segs{0, 2};
    std::vector<enc::HostCatIndexView> columns;
    columns.emplace_back(enc::CatStrArrayView{
        common::Span{offsets},
        common::Span{rank == 0 ? values_r0 : values_r1}});
    auto view = enc::HostColumnsView{common::Span{columns}, common::Span{segs}, 2};

    auto digest = data::HashCatHostContent(view);
    // mirror the packed-Allreduce shape used by quantile_dmatrix.{cc,cu} so a refactor
    // of either side is caught here too
    std::array<std::uint64_t, 2> hi{digest.primary, digest.secondary};
    std::array<std::uint64_t, 2> lo{digest.primary, digest.secondary};
    Context ctx;
    collective::SafeColl(
        collective::Allreduce(&ctx, linalg::MakeVec(lo.data(), 2), collective::Op::kMin));
    collective::SafeColl(
        collective::Allreduce(&ctx, linalg::MakeVec(hi.data(), 2), collective::Op::kMax));
    // independent primes + seeds ensure both streams diverge on this divergent input;
    // assert both to mirror the production pair of CHECK_EQ calls
    ASSERT_NE(lo[0], hi[0]);
    ASSERT_NE(lo[1], hi[1]);
  });
}

TEST(CatContainerHash, DistributedAgreeingDictsAllreduceMatch) {
  // positive sibling of DistributedDivergentDictsAllreduceDiffer: identical
  // dictionaries on every worker yield equal lo/hi after kMin+kMax Allreduce, so
  // the production CHECK_EQ pair does not fire
  constexpr std::int32_t kWorkers = 2;
  collective::TestDistributedGlobal(kWorkers, [] {
    std::vector<enc::CatCharT> values{'a', 'b', 'c', 'd', 'e'};
    std::vector<std::int32_t> offsets{0, 2, 5};
    std::vector<std::int32_t> segs{0, 2};
    std::vector<enc::HostCatIndexView> columns;
    columns.emplace_back(enc::CatStrArrayView{common::Span{offsets}, common::Span{values}});
    auto view = enc::HostColumnsView{common::Span{columns}, common::Span{segs}, 2};

    auto digest = data::HashCatHostContent(view);
    std::array<std::uint64_t, 2> hi{digest.primary, digest.secondary};
    std::array<std::uint64_t, 2> lo{digest.primary, digest.secondary};
    Context ctx;
    collective::SafeColl(
        collective::Allreduce(&ctx, linalg::MakeVec(lo.data(), 2), collective::Op::kMin));
    collective::SafeColl(
        collective::Allreduce(&ctx, linalg::MakeVec(hi.data(), 2), collective::Op::kMax));
    ASSERT_EQ(lo[0], hi[0]);
    ASSERT_EQ(lo[1], hi[1]);
  });
}
}  // namespace xgboost
