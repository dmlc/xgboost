#if defined(__CUDACC__)
#include "../../src/data/ellpack_page.cuh"
#endif

namespace xgboost {
#if defined(__CUDACC__)
namespace {
class HistogramCutsWrapper : public common::HistogramCuts {
 public:
  using SuperT = common::HistogramCuts;
  void SetValues(std::vector<float> cuts) {
    SuperT::cut_values_.HostVector() = std::move(cuts);
  }
  void SetPtrs(std::vector<uint32_t> ptrs) {
    SuperT::cut_ptrs_.HostVector() = std::move(ptrs);
  }
  void SetMins(std::vector<float> mins) {
    SuperT::min_vals_.HostVector() = std::move(mins);
  }
};
}  //  anonymous namespace

inline std::unique_ptr<EllpackPageImpl> BuildEllpackPage(
    int n_rows, int n_cols, bst_float sparsity= 0) {
  auto dmat = RandomDataGenerator(n_rows, n_cols, sparsity).Seed(3).GenerateDMatrix();
  const SparsePage& batch = *dmat->GetBatches<xgboost::SparsePage>().begin();

  HistogramCutsWrapper cmat;
  cmat.SetPtrs({0, 3, 6, 9, 12, 15, 18, 21, 24});
  // 24 cut fields, 3 cut fields for each feature (column).
  cmat.SetValues({0.30f, 0.67f, 1.64f,
          0.32f, 0.77f, 1.95f,
          0.29f, 0.70f, 1.80f,
          0.32f, 0.75f, 1.85f,
          0.18f, 0.59f, 1.69f,
          0.25f, 0.74f, 2.00f,
          0.26f, 0.74f, 1.98f,
          0.26f, 0.71f, 1.83f});
  cmat.SetMins({0.1f, 0.2f, 0.3f, 0.1f, 0.2f, 0.3f, 0.2f, 0.2f});

  bst_row_t row_stride = 0;
  const auto &offset_vec = batch.offset.ConstHostVector();
  for (size_t i = 1; i < offset_vec.size(); ++i) {
    row_stride = std::max(row_stride, offset_vec[i] - offset_vec[i-1]);
  }

  auto page = std::unique_ptr<EllpackPageImpl>(
      new EllpackPageImpl(0, cmat, batch, dmat->IsDense(), row_stride));

  return page;
}
#endif
}  // namespace xgboost
