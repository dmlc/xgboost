/*!
 * Copyright 2014 by Contributors
 * \file learner-inl.hpp
 * \brief learning algorithm
 * \author Tianqi Chen
 */
#ifndef XGBOOST_LEARNER_LEARNER_INL_HPP_
#define XGBOOST_LEARNER_LEARNER_INL_HPP_

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <limits>
#include "../sync/sync.h"
#include "../utils/io.h"
#include "./objective.h"
#include "./evaluation.h"
#include "../gbm/gbm.h"

namespace xgboost {
/*! \brief namespace for learning algorithm */
namespace learner {
/*!
 * \brief learner that takes do gradient boosting on specific objective functions
 *  and do training and prediction
 */
class BoostLearner : public rabit::Serializable {
 public:
  BoostLearner(void) {
    obj_ = NULL;
    gbm_ = NULL;
    name_obj_ = "reg:linear";
    name_gbm_ = "gbtree";
    silent = 0;
    prob_buffer_row = 1.0f;
    distributed_mode = 0;
    updater_mode = 0;
    pred_buffer_size = 0;
    seed_per_iteration = 0;
    seed = 0;
    save_base64 = 0;
  }
  virtual ~BoostLearner(void) {
    if (obj_ != NULL) delete obj_;
    if (gbm_ != NULL) delete gbm_;
  }
  /*!
   * \brief add internal cache space for mat, this can speedup prediction for matrix,
   *        please cache prediction for training and eval data
   *    warning: if the model is loaded from file from some previous training history
   *             set cache data must be called with exactly SAME
   *             data matrices to continue training otherwise it will cause error
   * \param mats array of pointers to matrix whose prediction result need to be cached
   */
  inline void SetCacheData(const std::vector<DMatrix*>& mats) {
    utils::Assert(cache_.size() == 0, "can only call cache data once");
    // assign buffer index
    size_t buffer_size = 0;
    for (size_t i = 0; i < mats.size(); ++i) {
      bool dupilicate = false;
      for (size_t j = 0; j < i; ++j) {
        if (mats[i] == mats[j]) dupilicate = true;
      }
      if (dupilicate) continue;
      // set mats[i]'s cache learner pointer to this
      mats[i]->cache_learner_ptr_ = this;
      cache_.push_back(CacheEntry(mats[i], buffer_size, mats[i]->info.num_row()));
      buffer_size += mats[i]->info.num_row();
    }
    char str_temp[25];
    utils::SPrintf(str_temp, sizeof(str_temp), "%lu",
                   static_cast<unsigned long>(buffer_size)); // NOLINT(*)
    this->SetParam("num_pbuffer", str_temp);
    this->pred_buffer_size = buffer_size;
  }
  /*!
   * \brief set parameters from outside
   * \param name name of the parameter
   * \param val  value of the parameter
   */
  inline void SetParam(const char *name, const char *val) {
    using namespace std;
    // in this version, bst: prefix is no longer required
    if (strncmp(name, "bst:", 4) != 0) {
      std::string n = "bst:"; n += name;
      this->SetParam(n.c_str(), val);
    }
    if (!strcmp(name, "silent")) silent = atoi(val);
    if (!strcmp(name, "dsplit")) {
      if (!strcmp(val, "col")) {
        this->SetParam("updater", "distcol");
        distributed_mode = 1;
      } else if (!strcmp(val, "row")) {
        this->SetParam("updater", "grow_histmaker,prune");
        distributed_mode = 2;
      } else {
        utils::Error("%s is invalid value for dsplit, should be row or col", val);
      }
    }
    if (!strcmp(name, "updater_mode")) updater_mode = atoi(val);
    if (!strcmp(name, "prob_buffer_row")) {
      prob_buffer_row = static_cast<float>(atof(val));
      utils::Check(distributed_mode == 0,
                   "prob_buffer_row can only be used in single node mode so far");
      this->SetParam("updater", "grow_colmaker,refresh,prune");
    }
    if (!strcmp(name, "eval_metric")) evaluator_.AddEval(val);
    if (!strcmp("seed", name)) {
      seed = atoi(val); random::Seed(seed);
    }
    if (!strcmp("seed_per_iter", name)) seed_per_iteration = atoi(val);
    if (!strcmp("save_base64", name)) save_base64 = atoi(val);
    if (!strcmp(name, "num_class")) {
      this->SetParam("num_output_group", val);
    }
    if (!strcmp(name, "nthread")) {
      omp_set_num_threads(atoi(val));
    }
    if (gbm_ == NULL) {
      if (!strcmp(name, "objective")) name_obj_ = val;
      if (!strcmp(name, "booster")) name_gbm_ = val;
      mparam.SetParam(name, val);
    }
    if (gbm_ != NULL) gbm_->SetParam(name, val);
    if (obj_ != NULL) obj_->SetParam(name, val);
    if (gbm_ == NULL || obj_ == NULL) {
      cfg_.push_back(std::make_pair(std::string(name), std::string(val)));
    }
  }
  // this is an internal function
  // initialize the trainer, called at InitModel and LoadModel
  inline void InitTrainer(bool calc_num_feature = true) {
    if (calc_num_feature) {
      // estimate feature bound
      unsigned num_feature = 0;
      for (size_t i = 0; i < cache_.size(); ++i) {
        num_feature = std::max(num_feature,
                               static_cast<unsigned>(cache_[i].mat_->info.num_col()));
      }
      // run allreduce on num_feature to find the maximum value
      rabit::Allreduce<rabit::op::Max>(&num_feature, 1);
      if (num_feature > mparam.num_feature) mparam.num_feature = num_feature;
    }
    char str_temp[25];
    utils::SPrintf(str_temp, sizeof(str_temp), "%d", mparam.num_feature);
    this->SetParam("bst:num_feature", str_temp);
  }
  /*!
   * \brief initialize the model
   */
  inline void InitModel(void) {
    this->InitTrainer();
    // initialize model
    this->InitObjGBM();
    // reset the base score
    mparam.base_score = obj_->ProbToMargin(mparam.base_score);
    // initialize GBM model
    gbm_->InitModel();
  }
  /*!
   * \brief load model from stream
   * \param fi input stream
   * \param calc_num_feature whether call InitTrainer with calc_num_feature
   */
  inline void LoadModel(utils::IStream &fi,  // NOLINT(*)
                        bool calc_num_feature = true) {
    utils::Check(fi.Read(&mparam, sizeof(ModelParam)) != 0,
                 "BoostLearner: wrong model format");
    {
      // backward compatibility code for compatible with old model type
      // for new model, Read(&name_obj_) is suffice
      uint64_t len;
      utils::Check(fi.Read(&len, sizeof(len)) != 0, "BoostLearner: wrong model format");
      if (len >= std::numeric_limits<unsigned>::max()) {
        int gap;
        utils::Check(fi.Read(&gap, sizeof(gap)) != 0, "BoostLearner: wrong model format");
        len = len >> static_cast<uint64_t>(32UL);
      }
      if (len != 0) {
        name_obj_.resize(len);
        utils::Check(fi.Read(&name_obj_[0], len) != 0, "BoostLearner: wrong model format");
      }
    }
    utils::Check(fi.Read(&name_gbm_), "BoostLearner: wrong model format");
    // delete existing gbm if any
    if (obj_ != NULL) delete obj_;
    if (gbm_ != NULL) delete gbm_;
    this->InitTrainer(calc_num_feature);
    this->InitObjGBM();
    char tmp[32];
    utils::SPrintf(tmp, sizeof(tmp), "%u", mparam.num_class);
    obj_->SetParam("num_class", tmp);
    gbm_->LoadModel(fi, mparam.saved_with_pbuffer != 0);
    if (mparam.saved_with_pbuffer == 0) {
      gbm_->ResetPredBuffer(pred_buffer_size);
    }
  }
  // rabit load model from rabit checkpoint
  virtual void Load(rabit::Stream *fi) {
    // for row split, we should not keep pbuffer
    this->LoadModel(*fi, false);
  }
  // rabit save model to rabit checkpoint
  virtual void Save(rabit::Stream *fo) const {
    // for row split, we should not keep pbuffer
    this->SaveModel(*fo, distributed_mode != 2);
  }
  /*!
   * \brief load model from file
   * \param fname file name
   */
  inline void LoadModel(const char *fname) {
    utils::IStream *fi = utils::IStream::Create(fname, "r");
    std::string header; header.resize(4);
    // check header for different binary encode
    // can be base64 or binary
    utils::Check(fi->Read(&header[0], 4) != 0, "invalid model");
    // base64 format
    if (header == "bs64") {
      utils::Base64InStream bsin(fi);
      bsin.InitPosition();
      this->LoadModel(bsin, true);
    } else if (header == "binf") {
      this->LoadModel(*fi, true);
    } else {
      delete fi;
      fi = utils::IStream::Create(fname, "r");
      this->LoadModel(*fi, true);
    }
    delete fi;
  }
  inline void SaveModel(utils::IStream &fo, bool with_pbuffer) const { // NOLINT(*)
    ModelParam p = mparam;
    p.saved_with_pbuffer = static_cast<int>(with_pbuffer);
    fo.Write(&p, sizeof(ModelParam));
    fo.Write(name_obj_);
    fo.Write(name_gbm_);
    gbm_->SaveModel(fo, with_pbuffer);
  }
  /*!
   * \brief save model into file
   * \param fname file name
   * \param with_pbuffer whether save pbuffer together
   */
  inline void SaveModel(const char *fname, bool with_pbuffer) const {
    utils::IStream *fo = utils::IStream::Create(fname, "w");
    if (save_base64 != 0 || !strcmp(fname, "stdout")) {
      fo->Write("bs64\t", 5);
      utils::Base64OutStream bout(fo);
      this->SaveModel(bout, with_pbuffer);
      bout.Finish('\n');
    } else {
      fo->Write("binf", 4);
      this->SaveModel(*fo, with_pbuffer);
    }
    delete fo;
  }
  /*!
   * \brief check if data matrix is ready to be used by training,
   *  if not intialize it
   * \param p_train pointer to the matrix used by training
   */
  inline void CheckInit(DMatrix *p_train) {
    int ncol = static_cast<int>(p_train->info.info.num_col);
    std::vector<bool> enabled(ncol, true);
    // set max row per batch to limited value
    // in distributed mode, use safe choice otherwise
    size_t max_row_perbatch = std::numeric_limits<size_t>::max();
    if (updater_mode != 0 || distributed_mode == 2) {
      max_row_perbatch = 32UL << 10UL;
    }
    // initialize column access
    p_train->fmat()->InitColAccess(enabled,
                                   prob_buffer_row,
                                   max_row_perbatch);
    const int kMagicPage = 0xffffab02;
    // check, if it is DMatrixPage, then use hist maker
    if (p_train->magic == kMagicPage) {
      this->SetParam("updater", "grow_histmaker,prune");
    }
  }
  /*!
   * \brief update the model for one iteration
   * \param iter current iteration number
   * \param p_train pointer to the data matrix
   */
  inline void UpdateOneIter(int iter, const DMatrix &train) {
    if (seed_per_iteration != 0 || rabit::IsDistributed()) {
      random::Seed(this->seed * kRandSeedMagic + iter);
    }
    this->PredictRaw(train, &preds_);
    obj_->GetGradient(preds_, train.info, iter, &gpair_);
    gbm_->DoBoost(train.fmat(), this->FindBufferOffset(train), train.info.info, &gpair_);
  }
  /*!
   * \brief whether model allow lazy checkpoint
   */
  inline bool AllowLazyCheckPoint(void) const {
    return gbm_->AllowLazyCheckPoint();
  }
  /*!
   * \brief evaluate the model for specific iteration
   * \param iter iteration number
   * \param evals datas i want to evaluate
   * \param evname name of each dataset
   * \return a string corresponding to the evaluation result
   */
  inline std::string EvalOneIter(int iter,
                                 const std::vector<const DMatrix*> &evals,
                                 const std::vector<std::string> &evname) {
    std::string res;
    char tmp[256];
    utils::SPrintf(tmp, sizeof(tmp), "[%d]", iter);
    res = tmp;
    for (size_t i = 0; i < evals.size(); ++i) {
      this->PredictRaw(*evals[i], &preds_);
      obj_->EvalTransform(&preds_);
      res += evaluator_.Eval(evname[i].c_str(), preds_, evals[i]->info, distributed_mode == 2);
    }
    return res;
  }
  /*!
   * \brief simple evaluation function, with a specified metric
   * \param data input data
   * \param metric name of metric
   * \return a pair of <evaluation name, result>
   */
  std::pair<std::string, float> Evaluate(const DMatrix &data, std::string metric) {
    if (metric == "auto") metric = obj_->DefaultEvalMetric();
    IEvaluator *ev = CreateEvaluator(metric.c_str());
    this->PredictRaw(data, &preds_);
    obj_->EvalTransform(&preds_);
    float res = ev->Eval(preds_, data.info);
    delete ev;
    return std::make_pair(metric, res);
  }
  /*!
   * \brief get prediction
   * \param data input data
   * \param output_margin whether to only predict margin value instead of transformed prediction
   * \param out_preds output vector that stores the prediction
   * \param ntree_limit limit number of trees used for boosted tree
   *   predictor, when it equals 0, this means we are using all the trees
   */
  inline void Predict(const DMatrix &data,
                      bool output_margin,
                      std::vector<float> *out_preds,
                      unsigned ntree_limit = 0,
                      bool pred_leaf = false) const {
    if (pred_leaf) {
      gbm_->PredictLeaf(data.fmat(), data.info.info, out_preds, ntree_limit);
    } else {
      this->PredictRaw(data, out_preds, ntree_limit);
      if (!output_margin) {
        obj_->PredTransform(out_preds);
      }
    }
  }
  /*!
   * \brief online prediction funciton, predict score for one instance at a time
   *  NOTE: use the batch prediction interface if possible, batch prediction is usually
   *        more efficient than online prediction
   *        This function is NOT threadsafe, make sure you only call from one thread
   *
   * \param inst the instance you want to predict
   * \param output_margin whether to only predict margin value instead of transformed prediction
   * \param out_preds output vector to hold the predictions
   * \param ntree_limit limit the number of trees used in prediction
   * \param root_index the root index
   * \sa Predict
   */
  inline void Predict(const SparseBatch::Inst &inst,
                      bool output_margin,
                      std::vector<float> *out_preds,
                      unsigned ntree_limit = 0) const {
    gbm_->Predict(inst, out_preds, ntree_limit);
    if (out_preds->size() == 1) {
      (*out_preds)[0] += mparam.base_score;
    }
    if (!output_margin) {
      obj_->PredTransform(out_preds);
    }
  }
  /*! \brief dump model out */
  inline std::vector<std::string> DumpModel(const utils::FeatMap& fmap, int option) {
    return gbm_->DumpModel(fmap, option);
  }

 protected:
  /*!
   * \brief initialize the objective function and GBM,
   * if not yet done
   */
  inline void InitObjGBM(void) {
    if (obj_ != NULL) return;
    utils::Assert(gbm_ == NULL, "GBM and obj should be NULL");
    obj_ = CreateObjFunction(name_obj_.c_str());
    gbm_ = gbm::CreateGradBooster(name_gbm_.c_str());
    this->InitAdditionDefaultParam();
    // set parameters
    for (size_t i = 0; i < cfg_.size(); ++i) {
      obj_->SetParam(cfg_[i].first.c_str(), cfg_[i].second.c_str());
      gbm_->SetParam(cfg_[i].first.c_str(), cfg_[i].second.c_str());
    }
    if (evaluator_.Size() == 0) {
      evaluator_.AddEval(obj_->DefaultEvalMetric());
    }
  }
  /*!
   * \brief additional default value for specific objs
   */
  inline void InitAdditionDefaultParam(void) {
    if (name_obj_ == "count:poisson") {
      obj_->SetParam("max_delta_step", "0.7");
      gbm_->SetParam("max_delta_step", "0.7");
    }
  }
  /*!
   * \brief get un-transformed prediction
   * \param data training data matrix
   * \param out_preds output vector that stores the prediction
   * \param ntree_limit limit number of trees used for boosted tree
   *   predictor, when it equals 0, this means we are using all the trees
   */
  inline void PredictRaw(const DMatrix &data,
                         std::vector<float> *out_preds,
                         unsigned ntree_limit = 0) const {
    gbm_->Predict(data.fmat(), this->FindBufferOffset(data),
                  data.info.info, out_preds, ntree_limit);
    // add base margin
    std::vector<float> &preds = *out_preds;
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(preds.size());
    if (data.info.base_margin.size() != 0) {
      utils::Check(preds.size() == data.info.base_margin.size(),
                   "base_margin.size does not match with prediction size");
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint j = 0; j < ndata; ++j) {
        preds[j] += data.info.base_margin[j];
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint j = 0; j < ndata; ++j) {
        preds[j] += mparam.base_score;
      }
    }
  }

  /*! \brief training parameter for regression */
  struct ModelParam{
    /* \brief global bias */
    float base_score;
    /* \brief number of features  */
    unsigned num_feature;
    /* \brief number of class, if it is multi-class classification  */
    int num_class;
    /*! \brief whether the model itself is saved with pbuffer */
    int saved_with_pbuffer;
    /*! \brief reserved field */
    int reserved[30];
    /*! \brief constructor */
    ModelParam(void) {
      std::memset(this, 0, sizeof(ModelParam));
      base_score = 0.5f;
      num_feature = 0;
      num_class = 0;
      saved_with_pbuffer = 0;
    }
    /*!
     * \brief set parameters from outside
     * \param name name of the parameter
     * \param val value of the parameter
     */
    inline void SetParam(const char *name, const char *val) {
      using namespace std;
      if (!strcmp("base_score", name)) base_score = static_cast<float>(atof(val));
      if (!strcmp("num_class", name)) num_class = atoi(val);
      if (!strcmp("bst:num_feature", name)) num_feature = atoi(val);
    }
  };
  // data fields
  // stored random seed
  int seed;
  // whether seed the PRNG each iteration
  // this is important for restart from existing iterations
  // default set to no, but will auto switch on in distributed mode
  int seed_per_iteration;
  // save model in base64 encoding
  int save_base64;
  // silent during training
  int silent;
  // distributed learning mode, if any, 0:none, 1:col, 2:row
  int distributed_mode;
  // updater mode, 0:normal, reserved for internal test
  int updater_mode;
  // cached size of predict buffer
  size_t pred_buffer_size;
  // maximum buffred row value
  float prob_buffer_row;
  // evaluation set
  EvalSet evaluator_;
  // model parameter
  ModelParam  mparam;
  // gbm model that back everything
  gbm::IGradBooster *gbm_;
  // name of gbm model used for training
  std::string name_gbm_;
  // objective fnction
  IObjFunction *obj_;
  // name of objective function
  std::string name_obj_;
  // configurations
  std::vector< std::pair<std::string, std::string> > cfg_;
  // temporal storages for prediciton
  std::vector<float> preds_;
  // gradient pairs
  std::vector<bst_gpair> gpair_;

 protected:
  // magic number to transform random seed
  static const int kRandSeedMagic = 127;
  // cache entry object that helps handle feature caching
  struct CacheEntry {
    const DMatrix *mat_;
    size_t buffer_offset_;
    size_t num_row_;
    CacheEntry(const DMatrix *mat, size_t buffer_offset, size_t num_row)
        :mat_(mat), buffer_offset_(buffer_offset), num_row_(num_row) {}
  };
  // find internal bufer offset for certain matrix, if not exist, return -1
  inline int64_t FindBufferOffset(const DMatrix &mat) const {
    for (size_t i = 0; i < cache_.size(); ++i) {
      if (cache_[i].mat_ == &mat && mat.cache_learner_ptr_ == this) {
        if (cache_[i].num_row_ == mat.info.num_row()) {
          return static_cast<int64_t>(cache_[i].buffer_offset_);
        }
      }
    }
    return -1;
  }
  // data structure field
  /*! \brief the entries indicates that we have internal prediction cache */
  std::vector<CacheEntry> cache_;
};
}  // namespace learner
}  // namespace xgboost
#endif  // XGBOOST_LEARNER_LEARNER_INL_HPP_
