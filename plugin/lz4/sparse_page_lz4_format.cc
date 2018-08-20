/*!
 * Copyright (c) 2015 by Contributors
 * \file sparse_page_lz4_format.cc
 *  XGBoost Plugin to enable LZ4 compressed format on the external memory pages.
 */
#include <xgboost/data.h>
#include <xgboost/logging.h>
#include <dmlc/registry.h>
#include <dmlc/parameter.h>
#include <lz4.h>
#include <lz4hc.h>
#include "../../src/data/sparse_page_writer.h"

namespace xgboost {
namespace data {

DMLC_REGISTRY_FILE_TAG(sparse_page_lz4_format);

// array to help compression of decompression.
template<typename DType>
class CompressArray {
 public:
  // the data content.
  std::vector<DType> data;
  // Decompression helper
  // number of chunks
  inline int num_chunk() const {
    CHECK_GT(raw_chunks_.size(), 1);
    return static_cast<int>(raw_chunks_.size() - 1);
  }
  // raw bytes
  inline size_t RawBytes() const {
    return raw_chunks_.back() * sizeof(DType);
  }
  // encoded bytes
  inline size_t EncodedBytes() const {
    return encoded_chunks_.back() +
        (encoded_chunks_.size() + raw_chunks_.size()) * sizeof(bst_uint);
  }
  // load the array from file.
  inline void Read(dmlc::SeekStream* fi);
  // run decode on chunk_id
  inline void Decompress(int chunk_id);
  // Compression helper
  // initialize the compression chunks
  inline void InitCompressChunks(const std::vector<bst_uint>& chunk_ptr);
  // initialize the compression chunks
  inline void InitCompressChunks(size_t chunk_size, size_t max_nchunk);
  // run decode on chunk_id, level = -1 means default.
  inline void Compress(int chunk_id, bool use_hc);
  // save the output buffer into file.
  inline void Write(dmlc::Stream* fo);

 private:
  // the chunk split of the data, by number of elements
  std::vector<bst_uint> raw_chunks_;
  // the encoded chunk, by number of bytes
  std::vector<bst_uint> encoded_chunks_;
  // output buffer of compression.
  std::vector<std::string> out_buffer_;
  // input buffer of data.
  std::string in_buffer_;
};

template<typename DType>
inline void CompressArray<DType>::Read(dmlc::SeekStream* fi) {
  CHECK(fi->Read(&raw_chunks_));
  CHECK(fi->Read(&encoded_chunks_));
  size_t buffer_size = encoded_chunks_.back();
  in_buffer_.resize(buffer_size);
  CHECK_EQ(fi->Read(dmlc::BeginPtr(in_buffer_), buffer_size), buffer_size);
  data.resize(raw_chunks_.back());
}

template<typename DType>
inline void CompressArray<DType>::Decompress(int chunk_id) {
  int chunk_size = static_cast<int>(
      raw_chunks_[chunk_id + 1] - raw_chunks_[chunk_id]) * sizeof(DType);
  int encoded_size = static_cast<int>(
      encoded_chunks_[chunk_id + 1] - encoded_chunks_[chunk_id]);
  // decompress data
  int src_size = LZ4_decompress_fast(
      dmlc::BeginPtr(in_buffer_) + encoded_chunks_[chunk_id],
      reinterpret_cast<char*>(dmlc::BeginPtr(data) + raw_chunks_[chunk_id]),
      chunk_size);
  CHECK_EQ(encoded_size, src_size);
}

template<typename DType>
inline void CompressArray<DType>::InitCompressChunks(
    const std::vector<bst_uint>& chunk_ptr) {
  raw_chunks_ = chunk_ptr;
  CHECK_GE(raw_chunks_.size(), 2);
  out_buffer_.resize(raw_chunks_.size() - 1);
  for (size_t i = 0; i < out_buffer_.size(); ++i) {
    out_buffer_[i].resize(raw_chunks_[i + 1] - raw_chunks_[i]);
  }
}

template<typename DType>
inline void CompressArray<DType>::InitCompressChunks(size_t chunk_size, size_t max_nchunk) {
  raw_chunks_.clear();
  raw_chunks_.push_back(0);
  size_t min_chunk_size = data.size() / max_nchunk;
  chunk_size = std::max(min_chunk_size, chunk_size);
  size_t nstep = data.size() / chunk_size;
  for (size_t i = 0; i < nstep; ++i) {
    raw_chunks_.push_back(raw_chunks_.back() + chunk_size);
    CHECK_LE(raw_chunks_.back(), data.size());
  }
  if (nstep == 0) raw_chunks_.push_back(0);
  raw_chunks_.back() = data.size();
  CHECK_GE(raw_chunks_.size(), 2);
  out_buffer_.resize(raw_chunks_.size() - 1);
  for (size_t i = 0; i < out_buffer_.size(); ++i) {
    out_buffer_[i].resize(raw_chunks_[i + 1] - raw_chunks_[i]);
  }
}

template<typename DType>
inline void CompressArray<DType>::Compress(int chunk_id, bool use_hc) {
  CHECK_LT(static_cast<size_t>(chunk_id + 1), raw_chunks_.size());
  std::string& buf = out_buffer_[chunk_id];
  size_t raw_chunk_size = (raw_chunks_[chunk_id + 1] - raw_chunks_[chunk_id]) * sizeof(DType);
  int bound = LZ4_compressBound(raw_chunk_size);
  CHECK_NE(bound, 0);
  buf.resize(bound);
  int encoded_size;
  if (use_hc) {
    encoded_size = LZ4_compress_HC(
        reinterpret_cast<char*>(dmlc::BeginPtr(data) + raw_chunks_[chunk_id]),
        dmlc::BeginPtr(buf), raw_chunk_size, buf.length(), 9);
  } else {
    encoded_size = LZ4_compress_default(
        reinterpret_cast<char*>(dmlc::BeginPtr(data) + raw_chunks_[chunk_id]),
        dmlc::BeginPtr(buf), raw_chunk_size, buf.length());
  }
  CHECK_NE(encoded_size, 0);
  CHECK_LE(static_cast<size_t>(encoded_size), buf.length());
  buf.resize(encoded_size);
}

template<typename DType>
inline void CompressArray<DType>::Write(dmlc::Stream* fo) {
  encoded_chunks_.clear();
  encoded_chunks_.push_back(0);
  for (size_t i = 0; i < out_buffer_.size(); ++i) {
    encoded_chunks_.push_back(encoded_chunks_.back() + out_buffer_[i].length());
  }
  fo->Write(raw_chunks_);
  fo->Write(encoded_chunks_);
  for (const std::string& buf : out_buffer_) {
    fo->Write(dmlc::BeginPtr(buf), buf.length());
  }
}

template<typename StorageIndex>
class SparsePageLZ4Format : public SparsePageFormat {
 public:
  explicit SparsePageLZ4Format(bool use_lz4_hc)
      : use_lz4_hc_(use_lz4_hc) {
    raw_bytes_ = raw_bytes_value_ = raw_bytes_index_ = 0;
    encoded_bytes_value_ = encoded_bytes_index_ = 0;
    nthread_ = dmlc::GetEnv("XGBOOST_LZ4_DECODE_NTHREAD", 4);
    nthread_write_ = dmlc::GetEnv("XGBOOST_LZ4_COMPRESS_NTHREAD", 12);
  }
  virtual ~SparsePageLZ4Format() {
    size_t encoded_bytes = raw_bytes_ +  encoded_bytes_value_ + encoded_bytes_index_;
    raw_bytes_ += raw_bytes_value_ + raw_bytes_index_;
    if (raw_bytes_ != 0) {
      LOG(CONSOLE) << "raw_bytes=" << raw_bytes_
                   << ", encoded_bytes=" << encoded_bytes
                   << ", ratio=" << double(encoded_bytes) / raw_bytes_
                   << ", ratio-index=" << double(encoded_bytes_index_) /raw_bytes_index_
                   << ", ratio-value=" << double(encoded_bytes_value_) /raw_bytes_value_;
    }
  }

  bool Read(SparsePage* page, dmlc::SeekStream* fi) override {
    auto& offset_vec = page->offset.HostVector();
    auto& data_vec = page->data.HostVector();
    if (!fi->Read(&(offset_vec))) return false;
    CHECK_NE(offset_vec.size(), 0) << "Invalid SparsePage file";
    this->LoadIndexValue(fi);

    data_vec.resize(offset_vec.back());
    CHECK_EQ(index_.data.size(), value_.data.size());
    CHECK_EQ(index_.data.size(), data_vec.size());
    for (size_t i = 0; i < data_vec.size(); ++i) {
      data_vec[i] = Entry(index_.data[i] + min_index_, value_.data[i]);
    }
    return true;
  }

  bool Read(SparsePage* page,
            dmlc::SeekStream* fi,
            const std::vector<bst_uint>& sorted_index_set) override {
    if (!fi->Read(&disk_offset_)) return false;
    this->LoadIndexValue(fi);
    auto& offset_vec = page->offset.HostVector();
    auto& data_vec = page->data.HostVector();
    offset_vec.clear();
    offset_vec.push_back(0);
    for (bst_uint cid : sorted_index_set) {
      offset_vec.push_back(
          offset_vec.back() + disk_offset_[cid + 1] - disk_offset_[cid]);
    }
    data_vec.resize(offset_vec.back());
    CHECK_EQ(index_.data.size(), value_.data.size());
    CHECK_EQ(index_.data.size(), disk_offset_.back());

    for (size_t i = 0; i < sorted_index_set.size(); ++i) {
      bst_uint cid = sorted_index_set[i];
      size_t dst_begin = offset_vec[i];
      size_t src_begin = disk_offset_[cid];
      size_t num = disk_offset_[cid + 1] - disk_offset_[cid];
      for (size_t j = 0; j < num; ++j) {
        data_vec[dst_begin + j] = Entry(
            index_.data[src_begin + j] + min_index_, value_.data[src_begin + j]);
      }
    }
    return true;
  }

  void Write(const SparsePage& page, dmlc::Stream* fo) override {
    const auto& offset_vec = page.offset.HostVector();
    const auto& data_vec = page.data.HostVector();
    CHECK(offset_vec.size() != 0 && offset_vec[0] == 0);
    CHECK_EQ(offset_vec.back(), data_vec.size());
    fo->Write(offset_vec);
    min_index_ = page.base_rowid;
    fo->Write(&min_index_, sizeof(min_index_));
    index_.data.resize(data_vec.size());
    value_.data.resize(data_vec.size());

    for (size_t i = 0; i < data_vec.size(); ++i) {
      bst_uint idx = data_vec[i].index - min_index_;
      CHECK_LE(idx, static_cast<bst_uint>(std::numeric_limits<StorageIndex>::max()))
          << "The storage index is chosen to limited to smaller equal than "
          << std::numeric_limits<StorageIndex>::max()
          << "min_index=" << min_index_;
      index_.data[i] = static_cast<StorageIndex>(idx);
      value_.data[i] = data_vec[i].fvalue;
    }

    index_.InitCompressChunks(kChunkSize, kMaxChunk);
    value_.InitCompressChunks(kChunkSize, kMaxChunk);

    int nindex = index_.num_chunk();
    int nvalue = value_.num_chunk();
    int ntotal = nindex + nvalue;
    #pragma omp parallel for schedule(dynamic, 1)  num_threads(nthread_write_)
    for (int i = 0; i < ntotal; ++i) {
      if (i < nindex) {
        index_.Compress(i, use_lz4_hc_);
      } else {
        value_.Compress(i - nindex, use_lz4_hc_);
      }
    }
    index_.Write(fo);
    value_.Write(fo);
    // statistics
    raw_bytes_index_ += index_.RawBytes() * sizeof(bst_uint) / sizeof(StorageIndex);
    raw_bytes_value_ += value_.RawBytes();
    encoded_bytes_index_ += index_.EncodedBytes();
    encoded_bytes_value_ += value_.EncodedBytes();
    raw_bytes_ += offset_vec.size() * sizeof(size_t);
  }

  inline void LoadIndexValue(dmlc::SeekStream* fi) {
    fi->Read(&min_index_, sizeof(min_index_));
    index_.Read(fi);
    value_.Read(fi);

    int nindex = index_.num_chunk();
    int nvalue = value_.num_chunk();
    int ntotal = nindex + nvalue;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthread_)
    for (int i = 0; i < ntotal; ++i) {
      if (i < nindex) {
        index_.Decompress(i);
      } else {
        value_.Decompress(i - nindex);
      }
    }
  }

 private:
  // default chunk size.
  static const size_t kChunkSize = 64 << 10UL;
  // maximum chunk size.
  static const size_t kMaxChunk = 128;
  // bool whether use hc
  bool use_lz4_hc_;
  // number of threads
  int nthread_;
  // number of writing threads
  int nthread_write_;
  // raw bytes
  size_t raw_bytes_, raw_bytes_index_, raw_bytes_value_;
  // encoded bytes
  size_t encoded_bytes_index_, encoded_bytes_value_;
  /*! \brief minimum index value */
  uint32_t min_index_;
  /*! \brief external memory column offset */
  std::vector<size_t> disk_offset_;
  // internal index
  CompressArray<StorageIndex> index_;
  // value set.
  CompressArray<bst_float> value_;
};

XGBOOST_REGISTER_SPARSE_PAGE_FORMAT(lz4)
.describe("Apply LZ4 binary data compression for ext memory.")
.set_body([]() {
    return new SparsePageLZ4Format<bst_uint>(false);
  });

XGBOOST_REGISTER_SPARSE_PAGE_FORMAT(lz4hc)
.describe("Apply LZ4 binary data compression(high compression ratio) for ext memory.")
.set_body([]() {
    return new SparsePageLZ4Format<bst_uint>(true);
  });

XGBOOST_REGISTER_SPARSE_PAGE_FORMAT(lz4i16hc)
.describe("Apply LZ4 binary data compression(16 bit index mode) for ext memory.")
.set_body([]() {
    return new SparsePageLZ4Format<uint16_t>(true);
  });

}  // namespace data
}  // namespace xgboost
