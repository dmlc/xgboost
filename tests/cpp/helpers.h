/**
 * Copyright 2016-2024, XGBoost contributors
 */
#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/json.h>
#include <xgboost/learner.h>  // for LearnerModelParam
#include <xgboost/model.h>    // for Configurable

#include <cstdint>  // std::int32_t
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#include "../../src/collective/communicator-inl.h"  // for GetRank
#include "../../src/common/common.h"                // for AllVisibleGPUs
#endif  // defined(__CUDACC__)

#include "filesystem.h"  // dmlc::TemporaryDirectory
#include "xgboost/linalg.h"
#if !defined(_OPENMP)
#include <thread>
#endif

#if defined(__CUDACC__)
#define DeclareUnifiedTest(name) GPU ## name
#else
#define DeclareUnifiedTest(name) name
#endif

#if defined(__CUDACC__)
#define GPUIDX (common::AllVisibleGPUs() == 1 ? 0 : collective::GetRank())
#else
#define GPUIDX (-1)
#endif

#if defined(__CUDACC__)
#define DeclareUnifiedDistributedTest(name) MGPU ## name
#else
#define DeclareUnifiedDistributedTest(name) name
#endif

namespace xgboost {
class ObjFunction;
class Metric;
struct LearnerModelParam;
class GradientBooster;
}

template <typename Float>
Float RelError(Float l, Float r) {
  static_assert(std::is_floating_point<Float>::value);
  return std::abs(1.0f - l / r);
}

bool FileExists(const std::string& filename);

void CreateSimpleTestData(const std::string& filename);

// Create a libsvm format file with 3 entries per-row. `zero_based` specifies whether it's
// 0-based indexing.
void CreateBigTestData(const std::string& filename, size_t n_entries, bool zero_based = true);

void CreateTestCSV(std::string const& path, size_t rows, size_t cols);

void CheckObjFunction(std::unique_ptr<xgboost::ObjFunction> const& obj,
                      std::vector<xgboost::bst_float> preds,
                      std::vector<xgboost::bst_float> labels,
                      std::vector<xgboost::bst_float> weights,
                      std::vector<xgboost::bst_float> out_grad,
                      std::vector<xgboost::bst_float> out_hess);

xgboost::Json CheckConfigReloadImpl(xgboost::Configurable* const configurable,
                                    std::string name);

template <typename T>
xgboost::Json CheckConfigReload(std::unique_ptr<T> const& configurable,
                                std::string name = "") {
  return CheckConfigReloadImpl(dynamic_cast<xgboost::Configurable*>(configurable.get()),
                               name);
}

void CheckRankingObjFunction(std::unique_ptr<xgboost::ObjFunction> const& obj,
                             std::vector<xgboost::bst_float> preds,
                             std::vector<xgboost::bst_float> labels,
                             std::vector<xgboost::bst_float> weights,
                             std::vector<xgboost::bst_uint> groups,
                             std::vector<xgboost::bst_float> out_grad,
                             std::vector<xgboost::bst_float> out_hess);

xgboost::bst_float GetMetricEval(
  xgboost::Metric * metric,
  xgboost::HostDeviceVector<xgboost::bst_float> const& preds,
  std::vector<xgboost::bst_float> labels,
  std::vector<xgboost::bst_float> weights = std::vector<xgboost::bst_float>(),
  std::vector<xgboost::bst_uint> groups = std::vector<xgboost::bst_uint>(),
  xgboost::DataSplitMode data_split_Mode = xgboost::DataSplitMode::kRow);

double GetMultiMetricEval(xgboost::Metric* metric,
                          xgboost::HostDeviceVector<xgboost::bst_float> const& preds,
                          xgboost::linalg::Tensor<float, 2> const& labels,
                          std::vector<xgboost::bst_float> weights = {},
                          std::vector<xgboost::bst_uint> groups = {},
                          xgboost::DataSplitMode data_split_Mode = xgboost::DataSplitMode::kRow);

namespace xgboost {

float GetBaseScore(Json const &config);

/*!
 * \brief Linear congruential generator.
 *
 * The distribution defined in std is not portable. Given the same seed, it
 * migth produce different outputs on different platforms or with different
 * compilers.  The SimpleLCG implemented here is to make sure all tests are
 * reproducible.
 */
class SimpleLCG {
 private:
  using StateType = uint64_t;
  static StateType constexpr kDefaultInit = 3;
  static StateType constexpr kDefaultAlpha = 61;
  static StateType constexpr kMaxValue = (static_cast<StateType>(1) << 32) - 1;

  StateType state_;
  StateType const alpha_;
  StateType const mod_;

 public:
  using result_type = StateType;  // NOLINT

 public:
  SimpleLCG() : state_{kDefaultInit}, alpha_{kDefaultAlpha}, mod_{kMaxValue} {}
  SimpleLCG(SimpleLCG const& that) = default;
  SimpleLCG(SimpleLCG&& that) = default;

  void Seed(StateType seed) { state_ = seed % mod_; }
  /*!
   * \brief Initialize SimpleLCG.
   *
   * \param state  Initial state, can also be considered as seed. If set to
   *               zero, SimpleLCG will use internal default value.
   */
  explicit SimpleLCG(StateType state)
      : state_{state == 0 ? kDefaultInit : state}, alpha_{kDefaultAlpha}, mod_{kMaxValue} {}

  StateType operator()();
  StateType Min() const;
  StateType Max() const;

  constexpr result_type static min() { return 0; };         // NOLINT
  constexpr result_type static max() { return kMaxValue; }  // NOLINT
};

template <typename ResultT>
class SimpleRealUniformDistribution {
 private:
  ResultT const lower_;
  ResultT const upper_;

  /*! \brief Over-simplified version of std::generate_canonical. */
  template <size_t Bits, typename GeneratorT>
  ResultT GenerateCanonical(GeneratorT* rng) const {
    static_assert(std::is_floating_point<ResultT>::value,
                  "Result type must be floating point.");
    long double const r = (static_cast<long double>(rng->Max())
                           - static_cast<long double>(rng->Min())) + 1.0L;
    auto const log2r = static_cast<size_t>(std::log(r) / std::log(2.0L));
    size_t m = std::max<size_t>(1UL, (Bits + log2r - 1UL) / log2r);
    ResultT sum_value = 0, r_k = 1;

    for (size_t k = m; k != 0; --k) {
      sum_value += static_cast<ResultT>((*rng)() - rng->Min()) * r_k;
      r_k *= static_cast<ResultT>(r);
    }

    ResultT res = sum_value / r_k;
    return res;
  }

 public:
  SimpleRealUniformDistribution(ResultT l, ResultT u) :
      lower_{l}, upper_{u} {}

  template <typename GeneratorT>
  ResultT operator()(GeneratorT* rng) const {
    ResultT tmp = GenerateCanonical<std::numeric_limits<ResultT>::digits,
                                    GeneratorT>(rng);
    auto ret = (tmp * (upper_ - lower_)) + lower_;
    // Correct floating point error.
    return std::max(ret, lower_);
  }
};

template <typename T>
Json GetArrayInterface(HostDeviceVector<T> const* storage, size_t rows, size_t cols) {
  Json array_interface{Object()};
  array_interface["data"] = std::vector<Json>(2);
  if (storage->DeviceCanRead()) {
    array_interface["data"][0] = Integer{reinterpret_cast<int64_t>(storage->ConstDevicePointer())};
    array_interface["stream"] = nullptr;
  } else {
    array_interface["data"][0] = Integer{reinterpret_cast<int64_t>(storage->ConstHostPointer())};
  }
  array_interface["data"][1] = Boolean(false);

  array_interface["shape"] = std::vector<Json>(2);
  array_interface["shape"][0] = rows;
  array_interface["shape"][1] = cols;

  char t = linalg::detail::ArrayInterfaceHandler::TypeChar<T>();
  array_interface["typestr"] = String(std::string{"<"} + t + std::to_string(sizeof(T)));
  array_interface["version"] = 3;
  return array_interface;
}

// Generate in-memory random data without using DMatrix.
class RandomDataGenerator {
  bst_idx_t rows_;
  size_t cols_;
  float sparsity_;

  float lower_{0.0f};
  float upper_{1.0f};

  bst_target_t n_targets_{1};

  DeviceOrd device_{DeviceOrd::CPU()};
  std::size_t n_batches_{0};
  std::uint64_t seed_{0};
  SimpleLCG lcg_;

  bst_bin_t bins_{0};
  std::vector<FeatureType> ft_;
  bst_cat_t max_cat_{32};
  bool on_host_{false};

  Json ArrayInterfaceImpl(HostDeviceVector<float>* storage, size_t rows, size_t cols) const;

  void GenerateLabels(std::shared_ptr<DMatrix> p_fmat) const;

 public:
  RandomDataGenerator(bst_idx_t rows, size_t cols, float sparsity)
      : rows_{rows}, cols_{cols}, sparsity_{sparsity}, lcg_{seed_} {}

  RandomDataGenerator& Lower(float v) {
    lower_ = v;
    return *this;
  }
  RandomDataGenerator& Upper(float v) {
    upper_ = v;
    return *this;
  }
  RandomDataGenerator& Device(DeviceOrd d) {
    device_ = d;
    return *this;
  }
  RandomDataGenerator& Batches(std::size_t n_batches) {
    n_batches_ = n_batches;
    return *this;
  }
  RandomDataGenerator& OnHost(bool on_host) {
    on_host_ = on_host;
    return *this;
  }
  RandomDataGenerator& Seed(uint64_t s) {
    seed_ = s;
    lcg_.Seed(seed_);
    return *this;
  }
  RandomDataGenerator& Bins(bst_bin_t b) {
    bins_ = b;
    return *this;
  }
  RandomDataGenerator& Type(common::Span<FeatureType> ft) {
    CHECK_EQ(ft.size(), cols_);
    ft_.resize(ft.size());
    std::copy(ft.cbegin(), ft.cend(), ft_.begin());
    return *this;
  }
  RandomDataGenerator& MaxCategory(bst_cat_t cat) {
    max_cat_ = cat;
    return *this;
  }
  RandomDataGenerator& Targets(bst_target_t n_targets) {
    n_targets_ = n_targets;
    return *this;
  }

  void GenerateDense(HostDeviceVector<float>* out) const;

  std::string GenerateArrayInterface(HostDeviceVector<float>* storage) const;

  /*!
   * \brief Generate batches of array interface stored in consecutive memory.
   *
   * \param storage The consecutive momory used to store the arrays.
   * \param batches Number of batches.
   *
   * \return A vector storing JSON string representation of interface for each batch, and
   *         a single JSON string representing the consecutive memory as a whole
   *         (combining all the batches).
   */
  std::pair<std::vector<std::string>, std::string> GenerateArrayInterfaceBatch(
      HostDeviceVector<float>* storage, size_t batches) const;

  std::string GenerateColumnarArrayInterface(std::vector<HostDeviceVector<float>>* data) const;

  void GenerateCSR(HostDeviceVector<float>* value, HostDeviceVector<std::size_t>* row_ptr,
                   HostDeviceVector<bst_feature_t>* columns) const;

  [[nodiscard]] std::shared_ptr<DMatrix> GenerateDMatrix(
      bool with_label = false, bool float_label = true, size_t classes = 1,
      DataSplitMode data_split_mode = DataSplitMode::kRow) const;

  [[nodiscard]] std::shared_ptr<DMatrix> GenerateSparsePageDMatrix(std::string prefix,
                                                                   bool with_label) const;

#if defined(XGBOOST_USE_CUDA)
  std::shared_ptr<DMatrix> GenerateDeviceDMatrix(bool with_label);
#endif
  std::shared_ptr<DMatrix> GenerateQuantileDMatrix(bool with_label);
};

// Generate an empty DMatrix, mostly for its meta info.
inline std::shared_ptr<DMatrix> EmptyDMatrix() {
  return RandomDataGenerator{0, 0, 0.0}.GenerateDMatrix();
}

inline std::vector<float> GenerateRandomCategoricalSingleColumn(int n, size_t num_categories) {
  std::vector<float> x(n);
  std::mt19937 rng(0);
  std::uniform_int_distribution<size_t> dist(0, num_categories - 1);
  std::generate(x.begin(), x.end(), [&]() { return static_cast<float>(dist(rng)); });
  // Make sure each category is present
  for (size_t i = 0; i < num_categories; i++) {
    x[i] = static_cast<decltype(x)::value_type>(i);
  }
  return x;
}

std::shared_ptr<DMatrix> GetDMatrixFromData(const std::vector<float>& x, std::size_t num_rows,
                                            bst_feature_t num_columns);

/**
 * \brief Create Sparse Page using data iterator.
 *
 * \param n_samples  Total number of rows for all batches combined.
 * \param n_features Number of features
 * \param n_batches  Number of batches
 * \param prefix     Cache prefix, can be used for specifying file path.
 *
 * \return A Sparse DMatrix with n_batches.
 */
std::unique_ptr<DMatrix> CreateSparsePageDMatrix(bst_idx_t n_samples, bst_feature_t n_features,
                                                 size_t n_batches, std::string prefix = "cache");

/**
 * Deprecated, stop using it
 */
std::unique_ptr<DMatrix> CreateSparsePageDMatrix(size_t n_entries, std::string prefix = "cache");

/**
 * Deprecated, stop using it
 *
 * \brief Creates dmatrix with some records, each record containing random number of
 *        features in [1, n_cols]
 *
 * \param n_rows      Number of records to create.
 * \param n_cols      Max number of features within that record.
 * \param page_size   Sparse page size for the pages within the dmatrix. If page size is 0
 *                    then the entire dmatrix is resident in memory; else, multiple sparse pages
 *                    of page size are created and backed to disk, which would have to be
 *                    streamed in at point of use.
 * \param deterministic The content inside the dmatrix is constant for this configuration, if true;
 *                      else, the content changes every time this method is invoked
 *
 * \return The new dmatrix.
 */
std::unique_ptr<DMatrix> CreateSparsePageDMatrixWithRC(
    size_t n_rows, size_t n_cols, size_t page_size, bool deterministic,
    const dmlc::TemporaryDirectory& tempdir = dmlc::TemporaryDirectory());

std::unique_ptr<GradientBooster> CreateTrainedGBM(std::string name, Args kwargs, size_t kRows,
                                                  size_t kCols,
                                                  LearnerModelParam const* learner_model_param,
                                                  Context const* generic_param);

/**
 * \brief Make a context that uses CUDA if device >= 0.
 */
inline Context MakeCUDACtx(std::int32_t device) {
  if (device == DeviceOrd::CPUOrdinal()) {
    return Context{};
  }
  return Context{}.MakeCUDA(device);
}

inline HostDeviceVector<GradientPair> GenerateRandomGradients(const size_t n_rows,
                                                              float lower = 0.0f,
                                                              float upper = 1.0f) {
  xgboost::SimpleLCG gen;
  xgboost::SimpleRealUniformDistribution<bst_float> dist(lower, upper);
  std::vector<GradientPair> h_gpair(n_rows);
  for (auto& gpair : h_gpair) {
    bst_float grad = dist(&gen);
    bst_float hess = dist(&gen);
    gpair = GradientPair(grad, hess);
  }
  HostDeviceVector<GradientPair> gpair(h_gpair);
  return gpair;
}

inline linalg::Matrix<GradientPair> GenerateRandomGradients(Context const* ctx, bst_idx_t n_rows,
                                                            bst_target_t n_targets,
                                                            float lower = 0.0f,
                                                            float upper = 1.0f) {
  auto g = GenerateRandomGradients(n_rows * n_targets, lower, upper);
  linalg::Matrix<GradientPair> gpair({n_rows, static_cast<bst_idx_t>(n_targets)}, ctx->Device());
  gpair.Data()->Copy(g);
  return gpair;
}

typedef void *DMatrixHandle;  // NOLINT(*);

class ArrayIterForTest {
 protected:
  HostDeviceVector<float> data_;
  size_t iter_{0};
  DMatrixHandle proxy_;
  std::unique_ptr<RandomDataGenerator> rng_;

  std::vector<std::string> batches_;
  std::string interface_;
  bst_idx_t rows_;
  size_t cols_;
  size_t n_batches_;

 public:
  bst_idx_t static constexpr Rows() { return 1024; }
  size_t static constexpr Batches() { return 100; }
  size_t static constexpr Cols() { return 13; }

 public:
  [[nodiscard]] std::string AsArray() const { return interface_; }

  virtual int Next() = 0;
  virtual void Reset() { iter_ = 0; }
  [[nodiscard]] std::size_t Iter() const { return iter_; }
  auto Proxy() -> decltype(proxy_) { return proxy_; }

  explicit ArrayIterForTest(float sparsity, bst_idx_t rows, size_t cols, size_t batches);
  /**
   * \brief Create iterator with user provided data.
   */
  explicit ArrayIterForTest(Context const* ctx, HostDeviceVector<float> const& data,
                            std::size_t n_samples, bst_feature_t n_features, std::size_t n_batches);
  virtual ~ArrayIterForTest();
};

class CudaArrayIterForTest : public ArrayIterForTest {
 public:
  explicit CudaArrayIterForTest(float sparsity, size_t rows = Rows(), size_t cols = Cols(),
                                size_t batches = Batches());
  int Next() override;
  ~CudaArrayIterForTest() override = default;
};

class NumpyArrayIterForTest : public ArrayIterForTest {
 public:
  explicit NumpyArrayIterForTest(float sparsity, bst_idx_t rows = Rows(), size_t cols = Cols(),
                                 size_t batches = Batches());
  explicit NumpyArrayIterForTest(Context const* ctx, HostDeviceVector<float> const& data,
                                 std::size_t n_samples, bst_feature_t n_features,
                                 std::size_t n_batches)
      : ArrayIterForTest{ctx, data, n_samples, n_features, n_batches} {}
  int Next() override;
  ~NumpyArrayIterForTest() override = default;
};

void DMatrixToCSR(DMatrix *dmat, std::vector<float> *p_data,
                  std::vector<size_t> *p_row_ptr,
                  std::vector<bst_feature_t> *p_cids);

typedef void *DataIterHandle;  // NOLINT(*)

inline void Reset(DataIterHandle self) {
  static_cast<ArrayIterForTest*>(self)->Reset();
}

inline int Next(DataIterHandle self) {
  return static_cast<ArrayIterForTest*>(self)->Next();
}

/**
 * @brief Create an array interface for host vector.
 */
template <typename T>
char const* Make1dInterfaceTest(T const* vec, std::size_t len) {
  static thread_local std::string str;
  str = linalg::Make1dInterface(vec, len);
  return str.c_str();
}

class RMMAllocator;
using RMMAllocatorPtr = std::unique_ptr<RMMAllocator, void(*)(RMMAllocator*)>;
RMMAllocatorPtr SetUpRMMResourceForCppTests(int argc, char** argv);

/*
 * \brief Make learner model param
 */
inline LearnerModelParam MakeMP(bst_feature_t n_features, float base_score, uint32_t n_groups,
                                DeviceOrd device = DeviceOrd::CPU()) {
  size_t shape[1]{1};
  LearnerModelParam mparam(n_features, linalg::Tensor<float, 1>{{base_score}, shape, device},
                           n_groups, 1, MultiStrategy::kOneOutputPerTree);
  return mparam;
}

inline std::int32_t AllThreadsForTest() { return Context{}.Threads(); }

inline DeviceOrd FstCU() { return DeviceOrd::CUDA(0); }

inline auto GMockThrow(StringView msg) {
  return ::testing::ThrowsMessage<dmlc::Error>(::testing::HasSubstr(msg));
}
}  // namespace xgboost
