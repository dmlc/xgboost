// this is a test case to test whether rabit can recover model when
// facing an exception
#include <rabit/rabit.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace rabit;

// dummy model
class Model : public rabit::Serializable {
 public:
  // iterations
  std::vector<float> data;
  // load from stream
  virtual void Load(rabit::Stream *fi) {
    fi->Read(&data);
  }
  /*! \brief save the model to the stream */
  virtual void Save(rabit::Stream *fo) const {
    fo->Write(data);
  }
  virtual void InitModel(size_t n) {
    data.clear();
    data.resize(n, 1.0f);
  }
};

inline void TestMax(Model *model, int ntrial, int iter) {
  int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();
  const int z = iter + 111;

  std::vector<float> ndata(model->data.size());
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (i * (rank+1)) % z  + model->data[i];
  }
  rabit::Allreduce<op::Max>(&ndata[0], ndata.size());

  for (size_t i = 0; i < ndata.size(); ++i) {
    float rmax = (i * 1) % z + model->data[i];
    for (int r = 0; r < nproc; ++r) {
      rmax = std::max(rmax, (float)((i * (r+1)) % z) + model->data[i]);
    }
    utils::Check(rmax == ndata[i], "[%d] TestMax check failurem i=%lu, rmax=%f, ndata=%f", rank, i, rmax, ndata[i]);
  }
}

inline void TestSum(Model *model, int ntrial, int iter) {
  int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();
  const int z = 131 + iter;

  std::vector<float> ndata(model->data.size());
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (i * (rank+1)) % z + model->data[i];
  }
  Allreduce<op::Sum>(&ndata[0], ndata.size());

  for (size_t i = 0; i < ndata.size(); ++i) {
    float rsum = model->data[i] * nproc;
    for (int r = 0; r < nproc; ++r) {
      rsum += (float)((i * (r+1)) % z);
    }
    utils::Check(fabsf(rsum - ndata[i]) < 1e-5 ,
                 "[%d] TestSum check failure, local=%g, allreduce=%g", rank, rsum, ndata[i]);
  }
  model->data = ndata;
}

inline void TestBcast(size_t n, int root, int ntrial, int iter) {
  int rank = rabit::GetRank();
  std::string s; s.resize(n);
  for (size_t i = 0; i < n; ++i) {
    s[i] = char(i % 126 + 1);
  }
  std::string res;
  if (root == rank) {
    res = s;
    rabit::Broadcast(&res, root);
  } else {
    rabit::Broadcast(&res, root);
  }
  utils::Check(res == s, "[%d] TestBcast fail", rank);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: <ndata> <config>\n");
    return 0;
  }
  int n = atoi(argv[1]);
  rabit::Init(argc, argv);
  int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();
  std::string name = rabit::GetProcessorName();
  Model model;
  srand(0);
  int ntrial = 0;
  for (int i = 1; i < argc; ++i) {
    int n;
    if (sscanf(argv[i], "rabit_num_trial=%d", &n) == 1) ntrial = n;
  }
  int iter = rabit::LoadCheckPoint(&model);
  if (iter == 0) {
    model.InitModel(n);
    printf("[%d] reload-trail=%d, init iter=%d\n", rank, ntrial, iter);
  } else {
    printf("[%d] reload-trail=%d, init iter=%d\n", rank, ntrial, iter);
  }
  for (int r = iter; r < 3; ++r) {
    TestMax(&model, ntrial, r);
    printf("[%d] !!!TestMax pass, iter=%d\n",  rank, r);
    int step = std::max(nproc / 3, 1);
    for (int i = 0; i < nproc; i += step) {
      TestBcast(n, i, ntrial, r);
    }
    printf("[%d] !!!TestBcast pass, iter=%d\n", rank, r);
    TestSum(&model, ntrial, r);
    printf("[%d] !!!TestSum pass, iter=%d\n", rank, r);
    rabit::LazyCheckPoint(&model);
    printf("[%d] !!!CheckPont pass, iter=%d\n", rank, r);
  }
  rabit::Finalize();
  return 0;
}
