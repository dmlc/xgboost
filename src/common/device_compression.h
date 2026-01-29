/**
 * Copyright 2025, XGBoost contributors
 *
 * @brief Implement (de)compression with the help of nvcomp and the HW decompression engine.
 */
#pragma once

#include <cstddef>  // for size_t
#include <numeric>  // for accumulate
#include <vector>   // for vector

#include "transform_iterator.h"  // for MakeIndexTransformIter

#if defined(XGBOOST_USE_NVCOMP)

#include <memory>  // for unique_ptr

#endif  // defined(XGBOOST_USE_NVCOMP)

namespace xgboost::dc {
/**
 * The cuda driver @ref CUmemDecompressParams struct without the pointers. We use this
 * struct to keep track of various buffer sizes. Naming of member variables follows the
 * CUDA struct.
 *
 * The src_nbytes stores the size of the allocated buffer for compressed data, and the
 * src_act_nbytes stores the actual size of the compressed data, which must be smaller
 * than the allocated size (src_nbytes). The nvcomp API over-allocate for compression.
 */
struct ComprParam {
  enum Algo {
    kLz4 = 0,
    kGDefalte = 1,
    kSnappy = 2,  // the only supported one at the moment.
  };

  // Compressed buffer bytes
  std::size_t src_nbytes = 0;
  // Actual compressed bytes
  std::size_t src_act_nbytes = 0;
  // Decompressed bytes.
  std::size_t dst_nbytes = 0;
  Algo algo;
};

/**
 * @brief A wrapper around vector of @ref ComprParam to help manage the chunks.
 */
struct CuMemParams {
  std::vector<ComprParam> params;

  CuMemParams() = default;
  CuMemParams(CuMemParams const& that) = default;
  CuMemParams(CuMemParams&& that) = default;
  CuMemParams& operator=(CuMemParams&& that) = default;
  CuMemParams& operator=(CuMemParams const& that) = default;

  explicit CuMemParams(std::size_t n_chunks) : params(n_chunks) {}

  ComprParam const& operator[](std::size_t i) const { return this->params[i]; }
  ComprParam& operator[](std::size_t i) { return this->params[i]; }
  ComprParam& at(std::size_t i) { return this->params.at(i); }              // NOLINT
  ComprParam const& at(std::size_t i) const { return this->params.at(i); }  // NOLINT
  void resize(std::size_t n) { this->params.resize(n); }                    // NOLINT

  [[nodiscard]] auto cbegin() const { return this->params.cbegin(); }  // NOLINT
  [[nodiscard]] auto cend() const { return this->params.cend(); }      // NOLINT

  [[nodiscard]] auto begin() const { return this->params.begin(); }  // NOLINT
  [[nodiscard]] auto end() const { return this->params.end(); }      // NOLINT
  [[nodiscard]] auto begin() { return this->params.begin(); }        // NOLINT
  [[nodiscard]] auto end() { return this->params.end(); }            // NOLINT

  [[nodiscard]] std::size_t size() const { return this->params.size(); }  // NOLINT
  [[nodiscard]] bool empty() const { return this->params.empty(); }       // NOLINT
  [[nodiscard]] auto data() const { return this->params.data(); }         // NOLINT

  [[nodiscard]] std::size_t TotalSrcBytes() const {
    auto it = common::MakeIndexTransformIter(
        [this](std::size_t i) { return this->params[i].src_nbytes; });
    return std::accumulate(it, it + this->size(), static_cast<std::size_t>(0));
  }
  [[nodiscard]] std::size_t TotalSrcActBytes() const {
    auto it = common::MakeIndexTransformIter(
        [this](std::size_t i) { return this->params[i].src_act_nbytes; });
    return std::accumulate(it, it + this->size(), static_cast<std::size_t>(0));
  }
  [[nodiscard]] std::size_t TotalDstBytes() const {
    auto it = common::MakeIndexTransformIter(
        [this](std::size_t i) { return this->params[i].dst_nbytes; });
    return std::accumulate(it, it + this->size(), static_cast<std::size_t>(0));
  }
};

class SnappyDecomprMgrImpl;

/**
 * @brief Help create and cache all decompression related meta data.
 *
 *   This struct is exposed to the CPU code. As a result, it's just a reference to the
 *   @SnappyDecomprMgrImpl .
 */
class SnappyDecomprMgr {
 public:
  SnappyDecomprMgr();
  SnappyDecomprMgr(SnappyDecomprMgr const& that) = delete;
  SnappyDecomprMgr(SnappyDecomprMgr&& that);
  SnappyDecomprMgr& operator=(SnappyDecomprMgr const& that) = delete;
  SnappyDecomprMgr& operator=(SnappyDecomprMgr&& that);

  ~SnappyDecomprMgr();

  SnappyDecomprMgrImpl* Impl() const;

  [[nodiscard]] bool Empty() const;
  /**
   * @brief The number of bytes of the uncompressed data.
   */
  [[nodiscard]] std::size_t DecompressedBytes() const;

 private:
  // Hide the CUDA API calls.
#if defined(XGBOOST_USE_NVCOMP)
  std::unique_ptr<SnappyDecomprMgrImpl> pimpl_;
#endif  // defined(XGBOOST_USE_NVCOMP)
};

struct DeStatus {
  bool avail{false};               // Whether the DE is present
  std::size_t max_output_size{0};  // Maximum output size of the buffer
};

// Get the query result of DE stored in a global variable.
[[nodiscard]] DeStatus const& GetGlobalDeStatus();
}  // namespace xgboost::dc
