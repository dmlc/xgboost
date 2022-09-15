/*!
 * Copyright 2016-2019 XGBoost contributors
 */
#ifndef XGBOOST_TESTS_CPP_HELPERS_H_
#define XGBOOST_TESTS_CPP_HELPERS_H_

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <xgboost/base.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/json.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../src/common/common.h"
#include "../../src/data/array_interface.h"
#include "../../src/gbm/gbtree_model.h"
#include "filesystem.h"  // dmlc::TemporaryDirectory

#if defined(__CUDACC__)
#define DeclareUnifiedTest(name) GPU ## name
#else
#define DeclareUnifiedTest(name) name
#endif

#if defined(__CUDACC__)
#define GPUIDX 0
#else
#define GPUIDX -1
#endif

namespace xgboost {
class ObjFunction;
class Metric;
struct LearnerModelParam;
class GradientBooster;
}

template <typename Float>
Float RelError(Float l, Float r) {
  static_assert(std::is_floating_point<Float>::value, "");
  return std::abs(1.0f - l / r);
}

bool FileExists(const std::string& filename);

int64_t GetFileSize(const std::string& filename);

void CreateSimpleTestData(const std::string& filename);

// Create a libsvm format file with 3 entries per-row. `zero_based` specifies whether it's
// 0-based indexing.
void CreateBigTestData(const std::string& filename, size_t n_entries, bool zero_based = true);

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
  std::vector<xgboost::bst_uint> groups = std::vector<xgboost::bst_uint>());

double GetMultiMetricEval(xgboost::Metric* metric,
                          xgboost::HostDeviceVector<xgboost::bst_float> const& preds,
                          xgboost::linalg::Tensor<float, 2> const& labels,
                          std::vector<xgboost::bst_float> weights = {},
                          std::vector<xgboost::bst_uint> groups = {});

namespace xgboost {
bool IsNear(std::vector<xgboost::bst_float>::const_iterator _beg1,
            std::vector<xgboost::bst_float>::const_iterator _end1,
            std::vector<xgboost::bst_float>::const_iterator _beg2);

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
      sum_value += ResultT((*rng)() - rng->Min()) * r_k;
      r_k *= r;
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
Json GetArrayInterface(HostDeviceVector<T> *storage, size_t rows, size_t cols) {
  Json array_interface{Object()};
  array_interface["data"] = std::vector<Json>(2);
  if (storage->DeviceCanRead()) {
    array_interface["data"][0] =
        Integer(reinterpret_cast<int64_t>(storage->ConstDevicePointer()));
    array_interface["stream"] = nullptr;
  } else {
    array_interface["data"][0] =
        Integer(reinterpret_cast<int64_t>(storage->ConstHostPointer()));
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
  bst_row_t rows_;
  size_t cols_;
  float sparsity_;

  float lower_;
  float upper_;

  int32_t device_;
  uint64_t seed_;
  SimpleLCG lcg_;

  size_t bins_;
  std::vector<FeatureType> ft_;
  bst_cat_t max_cat_;

  Json ArrayInterfaceImpl(HostDeviceVector<float> *storage, size_t rows,
                          size_t cols) const;

 public:
  RandomDataGenerator(bst_row_t rows, size_t cols, float sparsity)
      : rows_{rows}, cols_{cols}, sparsity_{sparsity}, lower_{0.0f}, upper_{1.0f},
        device_{-1}, seed_{0}, lcg_{seed_}, bins_{0} {}

  RandomDataGenerator &Lower(float v) {
    lower_ = v;
    return *this;
  }
  RandomDataGenerator& Upper(float v) {
    upper_ = v;
    return *this;
  }
  RandomDataGenerator& Device(int32_t d) {
    device_ = d;
    return *this;
  }
  RandomDataGenerator& Seed(uint64_t s) {
    seed_ = s;
    lcg_.Seed(seed_);
    return *this;
  }
  RandomDataGenerator& Bins(size_t b) {
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
  std::pair<std::vector<std::string>, std::string>
  GenerateArrayInterfaceBatch(HostDeviceVector<float> *storage,
                              size_t batches) const;

  std::string GenerateColumnarArrayInterface(
      std::vector<HostDeviceVector<float>> *data) const;

  void GenerateCSR(HostDeviceVector<float>* value, HostDeviceVector<bst_row_t>* row_ptr,
                   HostDeviceVector<bst_feature_t>* columns) const;

  std::shared_ptr<DMatrix> GenerateDMatrix(bool with_label = false,
                                           bool float_label = true,
                                           size_t classes = 1) const;
#if defined(XGBOOST_USE_CUDA)
  std::shared_ptr<DMatrix> GenerateDeviceDMatrix();
#endif
  std::shared_ptr<DMatrix> GenerateQuantileDMatrix();
};

inline std::vector<float>
GenerateRandomCategoricalSingleColumn(int n, size_t num_categories) {
  std::vector<float> x(n);
  std::mt19937 rng(0);
  std::uniform_int_distribution<size_t> dist(0, num_categories - 1);
  std::generate(x.begin(), x.end(), [&]() { return dist(rng); });
  // Make sure each category is present
  for(size_t i = 0; i < num_categories; i++) {
    x[i] = i;
  }
  return x;
}

std::shared_ptr<DMatrix> GetDMatrixFromData(const std::vector<float> &x,
                                            int num_rows, int num_columns);

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
std::unique_ptr<DMatrix> CreateSparsePageDMatrix(bst_row_t n_samples, bst_feature_t n_features,
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

gbm::GBTreeModel CreateTestModel(LearnerModelParam const* param, GenericParameter const* ctx,
                                 size_t n_classes = 1);

std::unique_ptr<GradientBooster> CreateTrainedGBM(
    std::string name, Args kwargs, size_t kRows, size_t kCols,
    LearnerModelParam const* learner_model_param,
    GenericParameter const* generic_param);

inline GenericParameter CreateEmptyGenericParam(int gpu_id) {
  xgboost::GenericParameter tparam;
  std::vector<std::pair<std::string, std::string>> args {
    {"gpu_id", std::to_string(gpu_id)}};
  tparam.Init(args);
  return tparam;
}

inline HostDeviceVector<GradientPair> GenerateRandomGradients(const size_t n_rows,
                                                              float lower= 0.0f, float upper = 1.0f) {
  xgboost::SimpleLCG gen;
  xgboost::SimpleRealUniformDistribution<bst_float> dist(lower, upper);
  std::vector<GradientPair> h_gpair(n_rows);
  for (auto &gpair : h_gpair) {
    bst_float grad = dist(&gen);
    bst_float hess = dist(&gen);
    gpair = GradientPair(grad, hess);
  }
  HostDeviceVector<GradientPair> gpair(h_gpair);
  return gpair;
}

typedef void *DMatrixHandle;  // NOLINT(*);

class ArrayIterForTest {
 protected:
  HostDeviceVector<float> data_;
  size_t iter_ {0};
  DMatrixHandle proxy_;
  std::unique_ptr<RandomDataGenerator> rng_;

  std::vector<std::string> batches_;
  std::string interface_;
  size_t rows_;
  size_t cols_;
  size_t n_batches_;

 public:
  size_t static constexpr Rows() { return 1024; }
  size_t static constexpr Batches() { return 100; }
  size_t static constexpr Cols() { return 13; }

 public:
  std::string AsArray() const { return interface_; }

  virtual int Next() = 0;
  virtual void Reset() { iter_ = 0; }
  size_t Iter() const { return iter_; }
  auto Proxy() -> decltype(proxy_) { return proxy_; }

  explicit ArrayIterForTest(float sparsity, size_t rows, size_t cols, size_t batches);
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
  explicit NumpyArrayIterForTest(float sparsity, size_t rows = Rows(), size_t cols = Cols(),
                                 size_t batches = Batches());
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

class RMMAllocator;
using RMMAllocatorPtr = std::unique_ptr<RMMAllocator, void(*)(RMMAllocator*)>;
RMMAllocatorPtr SetUpRMMResourceForCppTests(int argc, char** argv);

}  // namespace xgboost
#endif
