// this is a test case to test whether rabit can recover model when 
// facing an exception
#include <rabit.h>
#include <utils.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <fstream>
#include <ctime>

using namespace rabit;

class Model : public rabit::utils::ISerializable {
 public:
  std::vector<float> data;
  // load from stream
  virtual void Load(rabit::utils::IStream &fi) {
    fi.Read(&data);
  }
  /*! \brief save the model to the stream */
  virtual void Save(rabit::utils::IStream &fo) const {
    fo.Write(data);
  }
  virtual void InitModel(int k, int d) {
    data.resize(k * d + k, 0.0f);
  }
  
};

inline void KMeans(int ntrial, int iter, int k, int d, std::vector<float>& data, Model *model) {
  int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();

/*  const int z = iter + 111;

  std::vector<float> ndata(model->data.size());
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (i * (rank+1)) % z  + model->data[i];
  }
  rabit::Allreduce<op::Max>(&ndata[0], ndata.size());  
  if (ntrial == iter && rank == 3) {
    //throw MockException();
  }
  for (size_t i = 0; i < ndata.size(); ++i) {
    float rmax = (i * 1) % z + model->data[i];
    for (int r = 0; r < nproc; ++r) {
      rmax = std::max(rmax, (float)((i * (r+1)) % z) + model->data[i]);
    }
    utils::Check(rmax == ndata[i], "[%d] TestMax check failure\n", rank);
  }
  model->data = ndata;

*/
}

inline void ReadData(char* data_dir, int d, std::vector<std::vector<float> >* data) {
  int rank = rabit::GetRank();
  std::stringstream ss;
  ss << data_dir << rank;
  const char* file = ss.str().c_str();
  std::ifstream ifs(file);
  utils::Check(ifs.good(), "[%d] File %s does not exist\n", rank, file);
  float v = 0.0f;
  while(!ifs.eof()) {
    int i=0;
    std::vector<float> vec;
    while (i < d) {
      ifs >> v;
      vec.push_back(v);
      i++;
    }
    utils::Check(vec.size() % d == 0, "[%d] Invalid data size. %d instead of %d\n", rank, vec.size(), d);
    data->push_back(vec);
  }
}

inline void InitCentroids(int k, int d, std::vector<std::vector<float> >& data, Model* model) {
  int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();
  std::vector<std::vector<float> > candidate_centroids;
  candidate_centroids.resize(k, std::vector<float>(d));
  int elements = data.size();
  for (size_t i = 0; i < k; ++i) {
    int index = rand() % elements;
    candidate_centroids[i] = data[index];
  }
  for (size_t i = 0; i < k; ++i) {
    int proc = rand() % nproc;
    //utils::LogPrintf("[%d] proc=%d\n", rank, proc);
    std::string tmp_str;
    if (proc == rank) {
      std::ostringstream tmp;
      for (size_t j = 0; j < d ; ++j) {
        tmp << candidate_centroids[i][j];
        if (j != d-1) tmp << " ";
      }
      tmp_str = tmp.str();
      //utils::LogPrintf("[%d] centroid %s\n", rank, tmp_str.c_str());
      rabit::Bcast(&tmp_str, proc);
    } else {
      rabit::Bcast(&tmp_str, proc);
    }
    std::stringstream ss;
    ss.str(tmp_str);
    float val = 0.0f;
    int j = i * d;
    while(ss >> val) {
      model->data[j++] = val;
      //utils::LogPrintf("[%d] model[%d]=%.5f\n", rank, j-1, model->data[j-1]);
    }
    //count
    model->data[j] = 0;
    //utils::LogPrintf("[%d] model[375]=%.5f\n", rank, model->data[375]);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: <k> <d> <itr> <data_dir>\n");
    return 0;
  }
  int k = atoi(argv[1]);
  int d = atoi(argv[2]);
  int max_itr = atoi(argv[3]);

  rabit::Init(argc, argv);
  int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();
  std::string name = rabit::GetProcessorName();

  srand(0);
  int ntrial = 0;
  Model model;

  std::vector<std::vector<float> > data;
  int iter = rabit::LoadCheckPoint(&model);
  if (iter == 0) {
    ReadData(argv[4], d, &data);
    model.InitModel(k, d);
    InitCentroids(k, d, data, &model);
  } else {
    utils::LogPrintf("[%d] reload-trail=%d, init iter=%d\n", rank, ntrial, iter);
  }
  for (int r = iter; r < max_itr; ++r) { 
    //KMeans(ntrial, r, k, d, data, &model);
  }
  rabit::Finalize();
  return 0;
}
