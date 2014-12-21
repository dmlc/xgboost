// this is a test case to test whether rabit can recover model when 
// facing an exception
#include <rabit.h>
#include <rabit/utils.h>
#include "./toolkit_util.h"
#include <time.h>

using namespace rabit;

// kmeans model
class Model : public rabit::ISerializable {
 public:
  // matrix of centroids
  Matrix centroids;
  // load from stream
  virtual void Load(rabit::IStream &fi) {
    fi.Read(&centroids.nrow, sizeof(centroids.nrow));
    fi.Read(&centroids.ncol, sizeof(centroids.ncol));
    fi.Read(&centroids.data);
  }
  /*! \brief save the model to the stream */
  virtual void Save(rabit::IStream &fo) const {
    fo.Write(&centroids.nrow, sizeof(centroids.nrow));
    fo.Write(&centroids.ncol, sizeof(centroids.ncol));
    fo.Write(centroids.data);
  }
  virtual void InitModel(unsigned num_cluster, unsigned feat_dim) {
    centroids.Init(num_cluster, feat_dim);
  }
  // normalize L2 norm
  inline void Normalize(void) {
    for (size_t i = 0; i < centroids.nrow; ++i) {
      float *row = centroids[i];
      double wsum = 0.0;
      for (size_t j = 0; j < centroids.ncol; ++j) {
        wsum += row[j] * row[j];
      }
      wsum = sqrt(wsum);
      if (wsum < 1e-6) return;
      float winv = 1.0 / wsum;
      for (size_t j = 0; j < centroids.ncol; ++j) {
        row[j] *= winv;
      }
    }
  }
};
inline void InitCentroids(const SparseMat &data, Matrix *centroids) {
  int num_cluster = centroids->nrow; 
  for (int i = 0; i < num_cluster; ++i) {
    int index = Random(data.NumRow());
    SparseMat::Vector v = data[index];
    for (unsigned j = 0; j < v.length; ++j) {
      (*centroids)[i][v[j].findex] = v[j].fvalue;
    }
  }
  for (int i = 0; i < num_cluster; ++i) {
    int proc = Random(rabit::GetWorldSize());
    rabit::Broadcast((*centroids)[i], centroids->ncol * sizeof(float), proc);
  }
}

inline double Cos(const float *row,
                  const SparseMat::Vector &v) {
  double rdot = 0.0, rnorm = 0.0; 
  for (unsigned i = 0; i < v.length; ++i) {
    rdot += row[v[i].findex] * v[i].fvalue;
    rnorm += v[i].fvalue * v[i].fvalue;
  }
  return rdot  / sqrt(rnorm);
}
inline size_t GetCluster(const Matrix &centroids,
                         const SparseMat::Vector &v) {
  size_t imin = 0;
  double dmin = Cos(centroids[0], v);
  for (size_t k = 1; k < centroids.nrow; ++k) {
    double dist = Cos(centroids[k], v);
    if (dist > dmin) {
      dmin = dist; imin = k;
    }
  }
  return imin;
}
             
int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf("Usage: <data_dir> num_cluster max_iter <out_model>\n");
    return 0;
  }
  clock_t tStart = clock();

  srand(0);
  // load the data 
  SparseMat data;
  data.Load(argv[1]);
  // set the parameters
  int num_cluster = atoi(argv[2]);
  int max_iter = atoi(argv[3]);
  // intialize rabit engine
  rabit::Init(argc, argv);
  // load model
  Model model; 
  int iter = rabit::LoadCheckPoint(&model);
  if (iter == 0) {
    rabit::Allreduce<op::Max>(&data.feat_dim, 1);
    model.InitModel(num_cluster, data.feat_dim);
    InitCentroids(data, &model.centroids);
    model.Normalize();
    rabit::TrackerPrintf("[%d] start at %s\n",
                         rabit::GetRank(), rabit::GetProcessorName().c_str());
  } else {
    rabit::TrackerPrintf("[%d] restart iter=%d\n", rabit::GetRank(), iter);    
  }
  const unsigned num_feat = data.feat_dim;
  // matrix to store the result
  Matrix temp;
  for (int r = iter; r < max_iter; ++r) {    
    temp.Init(num_cluster, num_feat + 1, 0.0f);    
    auto lazy_get_centroid = [&]() {
      // lambda function used to calculate the data if necessary
      // this function may not be called when the result can be directly recovered
      const size_t ndata = data.NumRow();
      for (size_t i = 0; i < ndata; ++i) {
        SparseMat::Vector v = data[i];
        size_t k = GetCluster(model.centroids, v);
        // temp[k] += v
        for (size_t j = 0; j < v.length; ++j) {
          temp[k][v[j].findex] += v[j].fvalue;
        }
        // use last column to record counts
        temp[k][num_feat] += 1.0f;
      }
    };
    // call allreduce
    rabit::Allreduce<op::Sum>(&temp.data[0], temp.data.size(), lazy_get_centroid);
    // set number
    for (int k = 0; k < num_cluster; ++k) {
      float cnt = temp[k][num_feat];
      utils::Check(cnt != 0.0f, "get zero sized cluster");
      for (unsigned i = 0; i < num_feat; ++i) {
        model.centroids[k][i] = temp[k][i] / cnt;
      }
    }
    model.Normalize();
    rabit::CheckPoint(&model);
  }
  // output the model file to somewhere
  if (rabit::GetRank() == 0) {
    model.centroids.Print(argv[4]);
  }
  rabit::TrackerPrintf("[%d] Time taken: %f seconds\n", rabit::GetRank(), static_cast<float>(clock() - tStart) / CLOCKS_PER_SEC);
  rabit::Finalize();
  return 0;
}

