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

inline void KMeans(int ntrial, int iter, int k, int d, std::vector<std::vector<float> >& data, Model *model) {
  int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();

  utils::LogPrintf("[%d] Running KMeans iter=%d\n", rank, iter);

  // compute centroids
  std::vector<std::vector<float> > centroids;
  centroids.resize(k, std::vector<float>(d));
  for (int i = 0; i < k; ++i) {
    std::vector<float> centroid(d);
    int start = i * d + i;
    int count = model->data[start + d];
    //utils::LogPrintf("[%d] count=%d\n", rank, count);
    for (int j = start, l = 0; l < d; ++j, ++l) {
      centroid[l] = model->data[j] / count;
    }
    centroids[i] = centroid;
  }

  // compute assignments
  int size = data.size();
  std::vector<int> assignments(size, -1);
  for (int i = 0; i < size; ++i) {
    float max_sim = FLT_MIN;
    for (int j = 0; j < k; ++j) {
      float sim = utils::DotProduct(data[i], centroids[j]);
      if (sim > max_sim) {
        assignments[i] = j;
        max_sim = sim;
      }
    }
  }

  // add values and increment counts
  std::vector<float> ndata(k * d + k, 0.0f);
  for (int i=0; i < size; i++) {
    int index = assignments[i];
    int start = index * d + index;
    int j = start;
    for (int l = 0; l < d; ++j, ++l) {
      ndata[j] += data[i][l];
    }
    ndata[j] += 1;
  }

  // reduce
  rabit::Allreduce<op::Sum>(&ndata[0], ndata.size());
  model->data = ndata;

  /*
  if (rank == 0) {
    int counts = 0;
    for (int i = 0; i < k; ++i) {
      counts += model->data[i * d + i + d];
    }
    utils::LogPrintf("[%d] counts=%d\n", rank, counts);
  }
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
    std::vector<float> tmp(d, 0.0f);
    if (proc == rank) {
      tmp = candidate_centroids[i];
      rabit::Broadcast(&tmp, proc);
    } else {
      rabit::Broadcast(&tmp, proc);
    }
    int start = i * d + i;
    int j = start;
    for (int l = 0; l < d; ++j, ++l) {
      model->data[j] = tmp[l];
    }
    model->data[j] = 1;
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
