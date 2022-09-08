// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/data.h>

#include <future>
#include <thread>

#include "../../../src/common/io.h"
#include "../../../src/data/adapter.h"
#include "../../../src/data/file_iterator.h"
#include "../../../src/data/simple_dmatrix.h"
#include "../../../src/data/sparse_page_dmatrix.h"
#include "../filesystem.h"  // dmlc::TemporaryDirectory
#include "../helpers.h"

using namespace xgboost;  // NOLINT

template <typename Page>
void TestSparseDMatrixLoadFile() {
  dmlc::TemporaryDirectory tmpdir;
  auto opath = tmpdir.path + "/1-based.svm";
  CreateBigTestData(opath, 3 * 64, false);
  opath += "?indexing_mode=1";
  data::FileIterator iter{opath, 0, 1, "libsvm"};
  data::SparsePageDMatrix m{&iter,
                            iter.Proxy(),
                            data::fileiter::Reset,
                            data::fileiter::Next,
                            std::numeric_limits<float>::quiet_NaN(),
                            1,
                            tmpdir.path + "cache"};
  ASSERT_EQ(m.Info().num_col_, 5);
  ASSERT_EQ(m.Info().num_row_, 64);

  std::unique_ptr<dmlc::Parser<uint32_t>> parser(
      dmlc::Parser<uint32_t>::Create(opath.c_str(), 0, 1, "auto"));
  auto adapter = data::FileAdapter{parser.get()};

  data::SimpleDMatrix simple{&adapter, std::numeric_limits<float>::quiet_NaN(),
                             1};
  Page out;
  for (auto const& page : m.GetBatches<Page>()) {
    if (std::is_same<Page, SparsePage>::value) {
      out.Push(page);
    } else {
      out.PushCSC(page);
    }
  }
  ASSERT_EQ(m.Info().num_col_, simple.Info().num_col_);
  ASSERT_EQ(m.Info().num_row_, simple.Info().num_row_);

  for (auto const& page : simple.GetBatches<Page>()) {
    ASSERT_EQ(page.offset.HostVector(), out.offset.HostVector());
    for (size_t i = 0; i < page.data.Size(); ++i) {
      ASSERT_EQ(page.data.HostVector()[i].fvalue, out.data.HostVector()[i].fvalue);
    }
  }
}

TEST(SparsePageDMatrix, LoadFile) {
  TestSparseDMatrixLoadFile<SparsePage>();
  TestSparseDMatrixLoadFile<CSCPage>();
  TestSparseDMatrixLoadFile<SortedCSCPage>();
}

// allow caller to retain pages so they can process multiple pages at the same time.
template <typename Page>
void TestRetainPage() {
  auto m = CreateSparsePageDMatrix(10000);
  auto batches = m->GetBatches<Page>();
  auto begin = batches.begin();
  auto end = batches.end();

  std::vector<Page> pages;
  std::vector<std::shared_ptr<Page const>> iterators;
  for (auto it = begin; it != end; ++it) {
    iterators.push_back(it.Page());
    pages.emplace_back(Page{});
    if (std::is_same<Page, SparsePage>::value) {
      pages.back().Push(*it);
    } else {
      pages.back().PushCSC(*it);
    }
    ASSERT_EQ(pages.back().Size(), (*it).Size());
  }
  ASSERT_GE(iterators.size(), 2);

  for (size_t i = 0; i < iterators.size(); ++i) {
    ASSERT_EQ((*iterators[i]).Size(), pages.at(i).Size());
    ASSERT_EQ((*iterators[i]).data.HostVector(), pages.at(i).data.HostVector());
  }

  // make sure it's const and the caller can not modify the content of page.
  for (auto& page : m->GetBatches<Page>()) {
    static_assert(std::is_const<std::remove_reference_t<decltype(page)>>::value, "");
  }
}

TEST(SparsePageDMatrix, RetainSparsePage) {
  TestRetainPage<SparsePage>();
  TestRetainPage<CSCPage>();
  TestRetainPage<SortedCSCPage>();
}

TEST(SparsePageDMatrix, MetaInfo) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  size_t constexpr kEntries = 24;
  CreateBigTestData(tmp_file, kEntries);

  xgboost::DMatrix *dmat = xgboost::DMatrix::Load(
      tmp_file + "#" + tmp_file + ".cache", false, false);

  // Test the metadata that was parsed
  EXPECT_EQ(dmat->Info().num_row_, 8ul);
  EXPECT_EQ(dmat->Info().num_col_, 5ul);
  EXPECT_EQ(dmat->Info().num_nonzero_, kEntries);
  EXPECT_EQ(dmat->Info().labels.Size(), dmat->Info().num_row_);

  delete dmat;
}

TEST(SparsePageDMatrix, RowAccess) {
  std::unique_ptr<xgboost::DMatrix> dmat = xgboost::CreateSparsePageDMatrix(24);

  // Test the data read into the first row
  auto &batch = *dmat->GetBatches<xgboost::SparsePage>().begin();
  auto page = batch.GetView();
  auto first_row = page[0];
  ASSERT_EQ(first_row.size(), 3ul);
  EXPECT_EQ(first_row[2].index, 2u);
  EXPECT_NEAR(first_row[2].fvalue, 0.986566, 1e-4);
}

TEST(SparsePageDMatrix, ColAccess) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix *dmat =
      xgboost::DMatrix::Load(tmp_file + "#" + tmp_file + ".cache", true, false);

  // Loop over the batches and assert the data is as expected
  size_t iter = 0;
  for (auto const &col_batch : dmat->GetBatches<xgboost::SortedCSCPage>()) {
    auto col_page = col_batch.GetView();
    ASSERT_EQ(col_page.Size(), dmat->Info().num_col_);
    if (iter == 1) {
      ASSERT_EQ(col_page[0][0].fvalue, 0.f);
      ASSERT_EQ(col_page[3][0].fvalue, 30.f);
      ASSERT_EQ(col_page[3][0].index, 1);
      ASSERT_EQ(col_page[3].size(), 1);
    } else {
      ASSERT_EQ(col_page[1][0].fvalue, 10.0f);
      ASSERT_EQ(col_page[1].size(), 1);
    }
    CHECK_LE(col_batch.base_rowid, dmat->Info().num_row_);
    ++iter;
  }

  // Loop over the batches and assert the data is as expected
  iter = 0;
  for (auto const &col_batch : dmat->GetBatches<xgboost::CSCPage>()) {
    auto col_page = col_batch.GetView();
    EXPECT_EQ(col_page.Size(), dmat->Info().num_col_);
    if (iter == 0) {
      EXPECT_EQ(col_page[1][0].fvalue, 10.0f);
      EXPECT_EQ(col_page[1].size(), 1);
    } else {
      EXPECT_EQ(col_page[3][0].fvalue, 30.f);
      EXPECT_EQ(col_page[3].size(), 1);
    }
    iter++;
  }
  delete dmat;
}

TEST(SparsePageDMatrix, ThreadSafetyException) {
  size_t constexpr kEntriesPerCol = 3;
  size_t constexpr kEntries = 64 * kEntriesPerCol * 2;

  std::unique_ptr<xgboost::DMatrix> dmat =
      xgboost::CreateSparsePageDMatrix(kEntries);

  int threads = 1000;

  std::vector<std::future<void>> waiting;

  std::atomic<bool> exception {false};

  for (int32_t i = 0; i < threads; ++i) {
    waiting.emplace_back(std::async(std::launch::async, [&]() {
      try {
        auto iter = dmat->GetBatches<SparsePage>().begin();
        ++iter;
      } catch (...) {
        exception.store(true);
      }
    }));
  }

  using namespace std::chrono_literals;

  while (std::any_of(waiting.cbegin(), waiting.cend(), [](auto const &f) {
    return f.wait_for(0ms) != std::future_status::ready;
  })) {
    std::this_thread::sleep_for(50ms);
  }

  CHECK(exception);
}

// Multi-batches access
TEST(SparsePageDMatrix, ColAccessBatches) {
  size_t constexpr kPageSize = 1024, kEntriesPerCol = 3;
  size_t constexpr kEntries = kPageSize * kEntriesPerCol * 2;
  // Create multiple sparse pages
  std::unique_ptr<xgboost::DMatrix> dmat{xgboost::CreateSparsePageDMatrix(kEntries)};
  auto n_threads = omp_get_max_threads();
  omp_set_num_threads(16);
  for (auto const &page : dmat->GetBatches<xgboost::CSCPage>()) {
    ASSERT_EQ(dmat->Info().num_col_, page.Size());
  }
  omp_set_num_threads(n_threads);
}

auto TestSparsePageDMatrixDeterminism(int32_t threads) {
  omp_set_num_threads(threads);
  std::vector<float> sparse_data;
  std::vector<size_t> sparse_rptr;
  std::vector<bst_feature_t> sparse_cids;
  dmlc::TemporaryDirectory tempdir;
  std::string filename = tempdir.path + "/simple.libsvm";
  CreateBigTestData(filename, 1 << 16);

  data::FileIterator iter(filename, 0, 1, "auto");
  std::unique_ptr<DMatrix> sparse{new data::SparsePageDMatrix{
      &iter, iter.Proxy(), data::fileiter::Reset, data::fileiter::Next,
      std::numeric_limits<float>::quiet_NaN(), 1, filename}};

  DMatrixToCSR(sparse.get(), &sparse_data, &sparse_rptr, &sparse_cids);

  auto cache_name =
      data::MakeId(filename,
                   dynamic_cast<data::SparsePageDMatrix *>(sparse.get())) +
      ".row.page";
  std::string cache = common::LoadSequentialFile(cache_name);
  return cache;
}

TEST(SparsePageDMatrix, Determinism) {
#if defined(_MSC_VER)
  return;
#endif  // defined(_MSC_VER)
  std::vector<std::string> caches;
  for (size_t i = 1; i < 18; i += 2) {
    caches.emplace_back(TestSparsePageDMatrixDeterminism(i));
  }

  for (size_t i = 1; i < caches.size(); ++i) {
    ASSERT_EQ(caches[i], caches.front());
  }
}
