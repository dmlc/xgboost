#include "helpers.h"
#include "../../src/data/ellpack_page.cuh"

namespace xgboost {

std::unique_ptr<EllpackPageImpl> BuildEllpackPage(
    int n_rows, int n_cols, bst_float sparsity, bool on_device) {
  auto dmat = CreateDMatrix(n_rows, n_cols, sparsity, 3);
  const SparsePage& batch = *(*dmat)->GetBatches<xgboost::SparsePage>().begin();

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

  auto is_dense = (*dmat)->Info().num_nonzero_ ==
                  (*dmat)->Info().num_row_ * (*dmat)->Info().num_col_;
  size_t row_stride = 0;
  const auto &offset_vec = batch.offset.ConstHostVector();
  for (size_t i = 1; i < offset_vec.size(); ++i) {
    row_stride = std::max(row_stride, offset_vec[i] - offset_vec[i-1]);
  }

  if (on_device) {
    batch.offset.SetDevice(0);
    batch.offset.DeviceSpan();
    batch.data.SetDevice(0);
    batch.data.DeviceSpan();
  }

  auto page = std::unique_ptr<EllpackPageImpl>(new EllpackPageImpl(dmat->get()));
  page->InitCompressedData(0, cmat, row_stride, is_dense);
  page->CreateHistIndices(0, batch, RowStateOnDevice(batch.Size(), batch.Size()));

  delete dmat;

  return page;
}

}  // namespace xgboost