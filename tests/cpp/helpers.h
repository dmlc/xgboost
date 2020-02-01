/*!
 * Copyright 2016-2019 XGBoost contributors
 */
#ifndef XGBOOST_TESTS_CPP_HELPERS_H_
#define XGBOOST_TESTS_CPP_HELPERS_H_

#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include <gtest/gtest.h>

#include <dmlc/filesystem.h>
#include <xgboost/base.h>
#include <xgboost/objective.h>
#include <xgboost/metric.h>
#include <xgboost/json.h>
#include <xgboost/predictor.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/c_api.h>
#include <xgboost/learner.h>

#include "../../src/common/common.h"
#include "../../src/common/hist_util.h"
#if defined(__CUDACC__)
#include "../../src/data/ellpack_page.cuh"
#endif

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

bool FileExists(const std::string& filename);

int64_t GetFileSize(const std::string& filename);

void CreateSimpleTestData(const std::string& filename);

void CreateBigTestData(const std::string& filename, size_t n_entries);

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
  xgboost::HostDeviceVector<xgboost::bst_float> preds,
  std::vector<xgboost::bst_float> labels,
  std::vector<xgboost::bst_float> weights = std::vector<xgboost::bst_float> ());

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
  using StateType = int64_t;
  static StateType constexpr default_init_ = 3;
  static StateType constexpr default_alpha_ = 61;
  static StateType constexpr max_value_ = ((StateType)1 << 32) - 1;

  StateType state_;
  StateType const alpha_;
  StateType const mod_;

  StateType const seed_;

 public:
  SimpleLCG() : state_{default_init_},
                alpha_{default_alpha_}, mod_{max_value_}, seed_{state_}{}
  /*!
   * \brief Initialize SimpleLCG.
   *
   * \param state  Initial state, can also be considered as seed. If set to
   *               zero, SimpleLCG will use internal default value.
   * \param alpha  multiplier
   * \param mod    modulo
   */
  SimpleLCG(StateType state,
            StateType alpha=default_alpha_, StateType mod=max_value_)
      : state_{state == 0 ? default_init_ : state},
        alpha_{alpha}, mod_{mod} , seed_{state} {}

  StateType operator()();
  StateType Min() const;
  StateType Max() const;
};

template <typename ResultT>
class SimpleRealUniformDistribution {
 private:
  ResultT const lower;
  ResultT const upper;

  /*! \brief Over-simplified version of std::generate_canonical. */
  template <size_t Bits, typename GeneratorT>
  ResultT GenerateCanonical(GeneratorT* rng) const {
    static_assert(std::is_floating_point<ResultT>::value,
                  "Result type must be floating point.");
    long double const r = (static_cast<long double>(rng->Max())
                           - static_cast<long double>(rng->Min())) + 1.0L;
    size_t const log2r = std::log(r) / std::log(2.0L);
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
      lower{l}, upper{u} {}

  template <typename GeneratorT>
  ResultT operator()(GeneratorT* rng) const {
    ResultT tmp = GenerateCanonical<std::numeric_limits<ResultT>::digits,
                                    GeneratorT>(rng);
    return (tmp * (upper - lower)) + lower;
  }
};

/**
 * \fn  std::shared_ptr<xgboost::DMatrix> CreateDMatrix(int rows, int columns, float sparsity, int seed);
 *
 * \brief Creates dmatrix with uniform random data between 0-1.
 *
 * \param rows      The rows.
 * \param columns   The columns.
 * \param sparsity  The sparsity.
 * \param seed      The seed.
 *
 * \return  The new d matrix.
 */
std::shared_ptr<xgboost::DMatrix> *CreateDMatrix(int rows, int columns,
                                                 float sparsity, int seed = 0);

std::unique_ptr<DMatrix> CreateSparsePageDMatrix(
    size_t n_entries, size_t page_size, std::string tmp_file);

/**
 * \fn std::unique_ptr<DMatrix> CreateSparsePageDMatrixWithRC(size_t n_rows, size_t n_cols,
 *                                                            size_t page_size);
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

gbm::GBTreeModel CreateTestModel(LearnerModelParam const* param);

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

#if defined(__CUDACC__)
namespace {
class HistogramCutsWrapper : public common::HistogramCuts {
 public:
  using SuperT = common::HistogramCuts;
  void SetValues(std::vector<float> cuts) {
    SuperT::cut_values_ = std::move(cuts);
  }
  void SetPtrs(std::vector<uint32_t> ptrs) {
    SuperT::cut_ptrs_ = std::move(ptrs);
  }
  void SetMins(std::vector<float> mins) {
    SuperT::min_vals_ = std::move(mins);
  }
};
}  //  anonymous namespace

inline std::unique_ptr<EllpackPageImpl> BuildEllpackPage(
    int n_rows, int n_cols, bst_float sparsity= 0) {
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

  bst_row_t row_stride = 0;
  const auto &offset_vec = batch.offset.ConstHostVector();
  for (size_t i = 1; i < offset_vec.size(); ++i) {
    row_stride = std::max(row_stride, offset_vec[i] - offset_vec[i-1]);
  }

  auto page = std::unique_ptr<EllpackPageImpl>(new EllpackPageImpl(dmat->get(), {0, 256, 0}));
  page->InitInfo(0, (*dmat)->IsDense(), row_stride, cmat);
  page->InitCompressedData(0, n_rows);
  page->CreateHistIndices(0, batch, RowStateOnDevice(batch.Size(), batch.Size()));

  delete dmat;

  return page;
}
#endif

}  // namespace xgboost
#endif
