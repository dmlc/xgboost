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
#include <cfloat>

using namespace rabit;

class Model : public rabit::utils::ISerializable {
 public:
  std::vector<float> centroids;
  // load from stream
  virtual void Load(rabit::utils::IStream &fi) {
    fi.Read(&centroids);
  }
  /*! \brief save the model to the stream */
  virtual void Save(rabit::utils::IStream &fo) const {
    fo.Write(centroids);
  }
  virtual void InitModel(int k, int d) {
    centroids.resize(k * d, 0.0f);
  }
  
};

/*!\brief computes a random number modulo the value */
inline int Random(int value) {
  return rand() % value;
}

inline void KMeans(int ntrial, int iter, int k, int d, std::vector<std::vector<float> >& data, Model *model) {
  int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();

  utils::LogPrintf("[%d] Running KMeans iter=%d\n", rank, iter);

  // compute ndata based on assignments
  std::vector<float> ndata(k * d + k, 0.0f);
  for (int i = 0; i < data.size(); ++i) {
    float max_sim = FLT_MIN;
    int cindex = -1;
    for (int j = 0; j < k; ++j) {
      float sim = 0.0f;
      int cstart = j * d;
      for (int y = 0, z = cstart; y < d; ++y, ++z) {
        sim += model->centroids[z] * data[i][y];
      }
      if (sim > max_sim) {
        cindex = j;
        max_sim = sim;
      }
    }
    int start = cindex * d + cindex;
    int j = start;
    for (int l = 0; l < d; ++j, ++l) {
      ndata[j] += data[i][l];
    }
    // update count
    ndata[j] += 1;
  }

  // do Allreduce
  rabit::Allreduce<op::Sum>(&ndata[0], ndata.size());

  for (int i = 0; i < k; ++i) {
    int nstart = i * d + i;
    int cstart = i * d;
    int cend= cstart + d;
    int count = ndata[nstart + d];
    for (int j = nstart, l = cstart; l < cend; ++j, ++l) {
      model->centroids[l] = ndata[j] / count;
    }
  }
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
    int index = Random(elements);
    candidate_centroids[i] = data[index];
  }
  for (size_t i = 0; i < k; ++i) {
    int proc = Random(nproc);
    std::vector<float> tmp(d, 0.0f);
    if (proc == rank) {
      tmp = candidate_centroids[i];
      rabit::Broadcast(&tmp, proc);
    } else {
      rabit::Broadcast(&tmp, proc);
    }
    int start = i * d;
    int j = start;
    for (int l = 0; l < d; ++j, ++l) {
      model->centroids[j] = tmp[l];
    }
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
    KMeans(ntrial, r, k, d, data, &model);
    rabit::CheckPoint(model);
  }
  rabit::Finalize();
  return 0;
}
