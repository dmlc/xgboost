#include "hist_util.h"
#include "feature_bundling.h"

namespace xgboost {
namespace common {
void GHistIndexBlockMatrix::Build(const GHistIndexMatrix& gmat,
                                  const ColumnMatrix& colmat,
                                  const tree::TrainParam& param) {
  FeatureBundler bundler;

  std::vector<std::vector<unsigned>> groups = bundler.GroupFeatures(gmat, colmat, param);
  const size_t nrow = gmat.row_ptr.size() - 1;
  const auto nblock = static_cast<uint32_t>(groups.size());
  std::vector<size_t> index_size(nblock+1);
  index_size[0] = 0;

  const uint32_t nbins = gmat.cut.row_ptr.back();
  std::vector<uint32_t> bin2block(nbins);  // lookup table [bin id] => [block id]

  for (uint32_t group_id = 0; group_id < nblock; ++group_id) {
    auto const& group = groups[group_id];
    for (auto const& fid : group) {
      const uint32_t bin_begin = gmat.cut.row_ptr[fid];
      const uint32_t bin_end = gmat.cut.row_ptr[fid + 1];
      for (uint32_t bin_id = bin_begin; bin_id < bin_end; ++bin_id) {
        // many bins can map to same group
        bin2block[bin_id] = group_id;
      }
    }
  }

  // for (size_t rid = 0; rid < nrow; ++rid) {
  //   const size_t ibegin = gmat.row_ptr[rid];
  //   const size_t iend = gmat.row_ptr[rid + 1];

  //   // index_temp[block_id] needs iend - ibegin more
  //   for (size_t j = ibegin; j < iend; ++j) {
  //     const uint32_t bin_id = gmat.index[j];
  //     const uint32_t block_id = bin2block[bin_id];
  //     index_size[block_id+1] += 1;
  //   }
  // }
  size_t const n_indices = gmat.index.size();
  for (size_t i = 0; i < n_indices; ++i) {
    const uint32_t bin_id = gmat.index[i];
    const uint32_t block_id = bin2block[bin_id];
    index_size[block_id+1] += 1;
  }

  std::vector<size_t> indices_ptr = std::move(index_size);
  for (size_t i = 1; i < indices_ptr.size(); ++i) {
    indices_ptr[i] += indices_ptr[i-1];
  }
  std::vector<size_t> indices_count(nblock, 0);
  std::vector<size_t> row_ptr_count(nblock+1, 0);
  size_t row_ptr_step_size = nrow + 1;
  row_ptr_.resize(row_ptr_step_size * nblock);
  index_.resize(gmat.row_ptr.back());

  for (size_t rid = 0; rid < nrow; ++rid) {
    const size_t ibegin = gmat.row_ptr[rid];
    const size_t iend = gmat.row_ptr[rid + 1];

    for (size_t j = ibegin; j < iend; ++j) {
      const uint32_t bin_id = gmat.index[j];
      const uint32_t block_id = bin2block[bin_id];
      size_t index_begin = indices_ptr[block_id];
      index_[index_begin + indices_count[block_id]] = bin_id;
      indices_count[block_id] ++;
    }

    for (uint32_t block_id = 0; block_id < nblock; ++block_id) {
      size_t const begin = block_id * row_ptr_step_size;
      size_t const ind = begin + row_ptr_count[block_id+1];
      row_ptr_[ind] = indices_ptr[block_id+1] - indices_ptr[block_id];
      row_ptr_count[block_id+1] ++;
    }
  }

  for (size_t i = 1; i < row_ptr_count.size(); ++i) {
    row_ptr_count[i] = row_ptr_count[i+1];
  }

  blocks_.resize(nblock);
  for (uint32_t block_id = 0; block_id < nblock; ++block_id) {
    Block& blk = blocks_[block_id];
    blk.index_begin = &index_[indices_ptr[block_id]];
    blk.index_end = &index_[indices_ptr[block_id+1]];
    blk.row_ptr_begin = &row_ptr_[row_ptr_count[block_id]];
    blk.row_ptr_end = &row_ptr_[row_ptr_count[block_id+1]];
  }
}

}
}
