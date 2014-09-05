#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <ctime>
#include <string>
#include <cstring>
#include "io/io.h"
#include "utils/utils.h"
#include "utils/config.h"
#include "learner/learner-inl.hpp"

namespace xgboost {
/*!
 * \brief wrapping the training process 
 */
class BoostLearnTask{
 public:
  inline int Run(int argc, char *argv[]) {
    if (argc < 2) {
      printf("Usage: <config>\n");
      return 0;
    }
    utils::ConfigIterator itr(argv[1]);
    while (itr.Next()) {
      this->SetParam(itr.name(), itr.val());
    }
    for (int i = 2; i < argc; ++i) {
      char name[256], val[256];
      if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
        this->SetParam(name, val);
      }
    }
    this->InitData();
    this->InitLearner();
    if (task == "dump") {
      this->TaskDump(); return 0;
    }
    if (task == "eval") {
      this->TaskEval(); return 0;
    }
    if (task == "pred") {
      this->TaskPred();
    } else {
      this->TaskTrain();
    }
    return 0;
  }
  inline void SetParam(const char *name, const char *val) {
    if (!strcmp("silent", name)) silent = atoi(val);
    if (!strcmp("use_buffer", name)) use_buffer = atoi(val);
    if (!strcmp("num_round", name)) num_round = atoi(val);
    if (!strcmp("pred_margin", name)) pred_margin = atoi(val);
    if (!strcmp("ntree_limit", name)) ntree_limit = atoi(val);
    if (!strcmp("save_period", name)) save_period = atoi(val);
    if (!strcmp("eval_train", name)) eval_train = atoi(val);
    if (!strcmp("task", name)) task = val;
    if (!strcmp("data", name)) train_path = val;
    if (!strcmp("test:data", name)) test_path = val;
    if (!strcmp("model_in", name)) model_in = val;
    if (!strcmp("model_out", name)) model_out = val;
    if (!strcmp("model_dir", name)) model_dir_path = val;
    if (!strcmp("fmap", name)) name_fmap = val;
    if (!strcmp("name_dump", name)) name_dump = val;
    if (!strcmp("name_pred", name)) name_pred = val;
    if (!strcmp("dump_stats", name)) dump_model_stats = atoi(val);
    if (!strncmp("eval[", name, 5)) {
      char evname[256];
      utils::Assert(sscanf(name, "eval[%[^]]", evname) == 1, "must specify evaluation name for display");
      eval_data_names.push_back(std::string(evname));
      eval_data_paths.push_back(std::string(val));
    }
    learner.SetParam(name, val);
  }
 public:
  BoostLearnTask(void) {
    // default parameters
    silent = 0;
    use_buffer = 1;
    num_round = 10;
    save_period = 0;
    eval_train = 0;
    pred_margin = 0;
    ntree_limit = 0;
    dump_model_stats = 0;
    task = "train";
    model_in = "NULL";
    model_out = "NULL";
    name_fmap = "NULL";
    name_pred = "pred.txt";
    name_dump = "dump.txt";
    model_dir_path = "./";
    data = NULL;
  }
  ~BoostLearnTask(void){
    for (size_t i = 0; i < deval.size(); i++){
      delete deval[i];
    }
    if (data != NULL) delete data;
  }
 private:
  inline void InitData(void) {
    if (name_fmap != "NULL") fmap.LoadText(name_fmap.c_str());
    if (task == "dump") return;
    if (task == "pred") {
      data = io::LoadDataMatrix(test_path.c_str(), silent != 0, use_buffer != 0);
    } else {
      // training
      data = io::LoadDataMatrix(train_path.c_str(), silent != 0, use_buffer != 0);
      utils::Assert(eval_data_names.size() == eval_data_paths.size(), "BUG");
      for (size_t i = 0; i < eval_data_names.size(); ++i) {
        deval.push_back(io::LoadDataMatrix(eval_data_paths[i].c_str(), silent != 0, use_buffer != 0));
        devalall.push_back(deval.back());
      }
            
      std::vector<io::DataMatrix *> dcache(1, data);
      for (size_t i = 0; i < deval.size(); ++ i) {
        dcache.push_back(deval[i]);
      }
      // set cache data to be all training and evaluation data
      learner.SetCacheData(dcache);
      
      // add training set to evaluation set if needed
      if( eval_train != 0 ) {
        devalall.push_back(data);
        eval_data_names.push_back(std::string("train"));
      }
    }
  }
  inline void InitLearner(void) {
    if (model_in != "NULL"){
      utils::FileStream fi(utils::FopenCheck(model_in.c_str(), "rb"));
      learner.LoadModel(fi);
      fi.Close();
    } else {
      utils::Assert(task == "train", "model_in not specified");
      learner.InitModel();
    }
  }
  inline void TaskTrain(void) {
    const time_t start = time(NULL);
    unsigned long elapsed = 0;
    learner.CheckInit(data);
    for (int i = 0; i < num_round; ++i) {
      elapsed = (unsigned long)(time(NULL) - start);
      if (!silent) printf("boosting round %d, %lu sec elapsed\n", i, elapsed);
      learner.UpdateOneIter(i, *data); 
      std::string res = learner.EvalOneIter(i, devalall, eval_data_names);
      fprintf(stderr, "%s\n", res.c_str());
      if (save_period != 0 && (i + 1) % save_period == 0) {
        this->SaveModel(i);
      }
      elapsed = (unsigned long)(time(NULL) - start);
    }
    // always save final round
    if ((save_period == 0 || num_round % save_period != 0) && model_out != "NONE") {
      if (model_out == "NULL"){
        this->SaveModel(num_round - 1);
      } else {
        this->SaveModel(model_out.c_str());
      }
    }
    if (!silent){
      printf("\nupdating end, %lu sec in all\n", elapsed);
    }
  }
  inline void TaskEval(void) {
    learner.EvalOneIter(0, devalall, eval_data_names);
  }
  inline void TaskDump(void){
    FILE *fo = utils::FopenCheck(name_dump.c_str(), "w");
    std::vector<std::string> dump = learner.DumpModel(fmap, dump_model_stats != 0);
    for (size_t i = 0; i < dump.size(); ++ i) {
      fprintf(fo,"booster[%lu]:\n", i);
      fprintf(fo,"%s", dump[i].c_str()); 
    }
    fclose(fo);
  }
  inline void SaveModel(const char *fname) const {
    utils::FileStream fo(utils::FopenCheck(fname, "wb"));
    learner.SaveModel(fo);
    fo.Close();
  }
  inline void SaveModel(int i) const {
    char fname[256];
    sprintf(fname, "%s/%04d.model", model_dir_path.c_str(), i + 1);
    this->SaveModel(fname);
  }
  inline void TaskPred(void) {
    std::vector<float> preds;
    if (!silent) printf("start prediction...\n");
    learner.Predict(*data, pred_margin != 0, &preds, ntree_limit);
    if (!silent) printf("writing prediction to %s\n", name_pred.c_str());
    FILE *fo = utils::FopenCheck(name_pred.c_str(), "w");
    for (size_t i = 0; i < preds.size(); i++) {
      fprintf(fo, "%f\n", preds[i]);
    }
    fclose(fo);
  }
 private:
  /*! \brief whether silent */
  int silent;
  /*! \brief whether use auto binary buffer */
  int use_buffer;
  /*! \brief whether evaluate training statistics */            
  int eval_train;
  /*! \brief number of boosting iterations */
  int num_round;
  /*! \brief the period to save the model, 0 means only save the final round model */
  int save_period;
  /*! \brief the path of training/test data set */
  std::string train_path, test_path;
  /*! \brief the path of test model file, or file to restart training */
  std::string model_in;
  /*! \brief the path of final model file, to be saved */
  std::string model_out;
  /*! \brief the path of directory containing the saved models */
  std::string model_dir_path;
  /*! \brief task to perform */
  std::string task;
  /*! \brief name of predict file */
  std::string name_pred;
  /*!\brief limit number of trees in prediction */
  int ntree_limit;
  /*!\brief whether to directly output margin value */
  int pred_margin;
  /*! \brief whether dump statistics along with model */
  int dump_model_stats;
  /*! \brief name of feature map */
  std::string name_fmap;
  /*! \brief name of dump file */
  std::string name_dump;
  /*! \brief the paths of validation data sets */
  std::vector<std::string> eval_data_paths;
  /*! \brief the names of the evaluation data used in output log */
  std::vector<std::string> eval_data_names;
 private:
  io::DataMatrix* data;
  std::vector<io::DataMatrix*> deval;
  std::vector<const io::DataMatrix*> devalall;
  utils::FeatMap fmap;
  learner::BoostLearner learner;
};
}

int main(int argc, char *argv[]){
  xgboost::random::Seed(0);
  xgboost::BoostLearnTask tsk;
  return tsk.Run(argc, argv);
}
