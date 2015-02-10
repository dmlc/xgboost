#include "./linear.h"

namespace rabit {
namespace linear {
class LinearObjFunction : public solver::IObjFunction<float> {
 public:
  // training threads
  int nthread;
  // L2 regularization
  float reg_L2;
  // model
  LinearModel model;
  // training data
  SparseMat dtrain;
  // solver
  solver::LBFGSSolver<float> lbfgs;
  // constructor
  LinearObjFunction(void) {
    lbfgs.SetObjFunction(this);
    nthread = 1;
    reg_L2 = 0.0f;
    model.weight = NULL;
    task = "train";
    model_in = "NULL";
  }
  virtual ~LinearObjFunction(void) {
    if (model.weight != NULL) delete [] model.weight;
  }
  // set parameters
  inline void SetParam(const char *name, const char *val) {
    model.param.SetParam(name, val);
    lbfgs.SetParam(name, val);
    if (!strcmp(name, "num_feature")) {
      char ndigit[30];
      sprintf(ndigit, "%lu", model.param.num_feature + 1);
      lbfgs.SetParam("num_dim", ndigit);
    }
    if (!strcmp(name, "reg_L2")) {
      reg_L2 = static_cast<float>(atof(val));
    }
    if (!strcmp(name, "nthread")) {
      nthread = atoi(val);
    }
    if (!strcmp(name, "task")) task = val;
    if (!strcmp(name, "model_in")) model_in = val;
    if (!strcmp(name, "model_out")) model_out = val;
  }
  inline void Run(void) {
    if (model_in != "NULL") {
    }
    if (task == "train") {
      lbfgs.Run();
      
    } else if (task == "pred") {
      
    } else if (task == "eval") {
    } else {
      utils::Error("unknown task=%s", task.c_str());
    }
  }
  inline void LoadData(const char *fname) {
    dtrain.Load(fname);
  }
  virtual size_t InitNumDim(void)  {
    if (model_in == "NULL") {
      size_t ndim = dtrain.feat_dim;
      rabit::Allreduce<rabit::op::Max>(&ndim, 1);
      model.param.num_feature = std::max(ndim, model.param.num_feature);
    }
    return model.param.num_feature + 1;
  }
  virtual void InitModel(float *weight, size_t size) {
    if (model_in == "NULL") {
      memset(weight, 0.0f, size * sizeof(float));
      model.param.InitBaseScore();
    } else {
      rabit::Broadcast(model.weight, size * sizeof(float), 0);
      memcpy(weight, model.weight, size * sizeof(float));
    }
  }
  // load model
  virtual void Load(rabit::IStream &fi) {
    fi.Read(&model.param, sizeof(model.param));
  }
  virtual void Save(rabit::IStream &fo) const {
    fo.Write(&model.param, sizeof(model.param));
  }
  virtual double Eval(const float *weight, size_t size) {
   if (nthread != 0) omp_set_num_threads(nthread);
    utils::Check(size == model.param.num_feature + 1,
                 "size consistency check");
    double sum_val = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:sum_val)
    for (size_t i = 0; i < dtrain.NumRow(); ++i) {
      float py = model.param.PredictMargin(weight, dtrain[i]);
      float fv = model.param.MarginToLoss(dtrain.labels[i], py);
      sum_val += fv;
    }
    if (rabit::GetRank() == 0) {
      // only add regularization once
      if (reg_L2 != 0.0f) {
        double sum_sqr = 0.0;
        for (size_t i = 0; i < model.param.num_feature; ++i) {
          sum_sqr += weight[i] * weight[i];
        }
        sum_val += 0.5 * reg_L2 * sum_sqr;        
      }
    }
    utils::Check(!std::isnan(sum_val), "nan occurs");
    return sum_val;
  }
  virtual void CalcGrad(float *out_grad,
                        const float *weight,
                        size_t size) {
   if (nthread != 0) omp_set_num_threads(nthread);
   utils::Check(size == model.param.num_feature + 1,
                 "size consistency check");
    memset(out_grad, 0.0f, sizeof(float) * size);
    double sum_gbias = 0.0;    
    #pragma omp parallel for schedule(static) reduction(+:sum_gbias)
    for (size_t i = 0; i < dtrain.NumRow(); ++i) {
      SparseMat::Vector v = dtrain[i];
      float py = model.param.Predict(weight, v);
      float grad = model.param.PredToGrad(dtrain.labels[i], py);
      for (index_t j = 0; j < v.length; ++j) {
        out_grad[v[j].findex] += v[j].fvalue * grad;
      }
      sum_gbias += grad;
    }
    out_grad[model.param.num_feature] = static_cast<float>(sum_gbias);
    if (rabit::GetRank() == 0) {
      // only add regularization once
      if (reg_L2 != 0.0f) {
        for (size_t i = 0; i < model.param.num_feature; ++i) {
          out_grad[i] += reg_L2 * weight[i];
        }
      }
    }
  }

 private:
  std::string task;
  std::string model_in;
  std::string model_out;
};
}  // namespace linear
}  // namespace rabit

int main(int argc, char *argv[]) {
  if (argc < 2) {
    // intialize rabit engine
    rabit::Init(argc, argv);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Usage: <data_in> param=val\n");
    }
    rabit::Finalize();
    return 0;
  }
  rabit::linear::LinearObjFunction linear;
  if (!strcmp(argv[1], "stdin")) {
    linear.LoadData(argv[1]);
    rabit::Init(argc, argv);
  } else {
    rabit::Init(argc, argv);
    linear.LoadData(argv[1]);
  }
  for (int i = 2; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
      linear.SetParam(name, val);
    }
  }
  linear.Run();
  rabit::Finalize();
  return 0;
}
