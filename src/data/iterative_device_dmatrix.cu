#include "iterative_device_dmatrix.h"
#include "../common/hist_util.h"
#include "ellpack_page.cuh"
#include "device_adapter.cuh"
namespace xgboost {
namespace data {
template <typename Adapter>
IterativeDeviceDMatrix::IterativeDeviceDMatrix(Adapter* adapter, float missing,
                                               int nthread, int max_bin) {
  common::HistogramCuts cuts = common::AdapterDeviceSketch(adapter, max_bin, missing);
  auto device = adapter->DeviceIdx();
  p_ = BatchParam{device, max_bin};
  pages_.reset(new std::vector<EllpackPage>);

  adapter->BeforeFirst();
  while (adapter->Next()) {
    auto& batch = adapter->Value();
    auto rows = adapter->Value().NumRows();
    dh::caching_device_vector<size_t> row_counts(rows + 1, 0);
    common::Span<size_t> row_counts_span(row_counts.data().get(),
                                         row_counts.size());
    size_t row_stride =
        GetRowCounts(batch, row_counts_span, device, missing);
    pages_->emplace_back(EllpackPage());
    (*pages_->back().Impl()) =
        EllpackPageImpl(batch, missing, adapter->DeviceIdx(),
                        this->IsDense(), nthread,
                        adapter->IsRowMajor(),
                        row_counts_span,
                        row_stride,
                        rows,
                        cuts);

    MetaInfo info;
    if (batch.Labels() != nullptr) {
      info.SetInfo("label", batch.Labels(), DataType::kFloat32, batch.Size());
    }
    if (batch.Weights() != nullptr) {
      info.SetInfo("weight", batch.Weights(), DataType::kFloat32, batch.Size());
    }
    if (batch.BaseMargin() != nullptr) {
      info.SetInfo("base_margin", batch.BaseMargin(), DataType::kFloat32, batch.Size());
    }
    if (batch.Qid() != nullptr) {
      info.SetInfo("group", batch.Qid(), DataType::kFloat32, batch.Size());
    }

    this->info_.Append(info);
  }
}

template IterativeDeviceDMatrix::IterativeDeviceDMatrix<data::CudaArrayInterfaceCallbackAdapter>(
    data::CudaArrayInterfaceCallbackAdapter* adapter, float missing, int nthread, int max_bin);
}  // namespace data
}  // namespace xgboost