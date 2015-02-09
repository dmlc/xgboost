/*!
 *  Copyright (c) 2015 by Contributors
 * \file lbfgs.h
 * \brief L-BFGS solver for general optimization problem
 *
 * \author Tianqi Chen
 */
#ifndef RABIT_LBFGS_H_
#define RABIT_LBFGS_H_
#include <cmath>
#include <rabit.h>

namespace rabit {
/*! \brief namespace of solver for general problems */
namespace solver {
/*! \brief an L-BFGS solver */
template<typename DType>
class LBFGSSolver {
 public:
  LBFGSSolver(void) {
    reg_L1 = 0.0f;
    max_linesearch_iter = 1000;
    linesearch_backoff = 0.5f;
  }
  // initialize the L-BFGS solver
  inline void Init(size_t num_feature, size_t size_memory) {
    mdot.Init(size_memory_);
    hist.Init(num_feature, size_memory_);
  }

 protected:
  // find the delta value, given gradient
  // return dot(dir, l1grad)
  virtual double FindChangeDirection(DType *dir,
                                     const DType *grad,
                                     const DType *weight) {
    int m = static_cast<int>(size_memory_);
    int n = static_cast<int>(hist.num_useful());
    const DType *gsub = grad + range_begin_;
    const size_t nsub = range_end_ - range_begin_;
    double vdot;
    if (n != 0) {
      // hist[m + n - 1] stores old gradient
      Minus(hist[m + n - 1], gsub, hist[m + n - 1], nsub);
      SetL1Dir(hist[2 * m], gsub, weight + range_begin_, nsub);
      // index set for calculating results
      std::vector<std::pair<size_t, size_t> > idxset;
      for (int j = 0; j < n; ++j) {
        idxset.push_back(std::make_pair(j, 2 * m));
        idxset.push_back(std::make_pair(j, n - 1));
        idxset.push_back(std::make_pair(j, m + n - 1));
      }
      for (int j = 0; j < n; ++j) {
        idxset.push_back(std::make_pair(m + j, 2 * m));
        idxset.push_back(std::make_pair(m + j, m + n - 1));
      }
      // calculate dot products
      std::vector<double> tmp(idxset.size());
      for (size_t i = 0; i < tmp.size(); ++i) {
        tmp[i] = hist.CalcDot(idxset[i].first, idxset[i].second);
      }
      rabit::Allreduce<rabit::op::Sum>(BeginPtr(tmp), tmp.size());
      for (size_t i = 0; i < tmp.size(); ++i) {
        mdot.Get(idxset[i].first, idxset[i].second) = tmp[i];
      }
      // BFGS steps
      std::vector<double> alpha(n);
      std::vector<double> delta(2 * n + 1, 0.0);
      delta[2 * n] = 1.0;
      // backward step
      for (int j = n - 1; j >= 0; --j) {
        double vsum = 0.0;
        for (size_t k = 0; k < delta.size(); ++k) {
          vsum += delta[k] * mdot.Get(k, j);
        }
        alpha[j] = vsum / mdot.Get(j, m + j);
        delta[m + j] = delta[m + j] - alpha[j];
      }
      // scale
      double scale = mdot.Get(n - 1, m + n - 1) /
      mdot.Get(m + n - 1, m + n - 1);
      for (size_t k = 0; k < delta.size(); ++k) {
        delta[k] *= scale;
      }
      // forward step
      for (int j = 0; j < n; ++j) {
        double vsum = 0.0;
        for (size_t k = 0; k < delta.size(); ++k) {
          vsum += delta[k] * mdot.Get(k, m + j);
        }
        double beta = vsum / mdot.Get(j, m + j);
        delta[j] = delta[j] + (alpha[j] - beta);
      }
      // set all to zero
      std::fill(dir, dir + num_feature_, 0.0f);
      DType *dirsub = dir + range_begin_; 
      for (int i = 0; i < n; ++i) {
        AddScale(dirsub, dirsub, hist[i], delta[i], nsub);
        AddScale(dirsub, dirsub, hist[m + i], delta[m + i], nsub);
      }
      AddScale(dirsub, dirsub, hist[2 * m], delta[2 * m], nsub);
      FixDirL1Sign(dir + range_begin_, hist[2 * m], nsub);
      vdot = -Dot(dir + range_begin_, hist[2 * m], nsub);
      // allreduce to get full direction
      rabit::Allreduce<rabit::op::Sum>(dir, num_feature_);
      rabit::Allreduce<rabit::op::Sum>(&vdot, 1);
    } else {
      SetL1Dir(dir, grad, weight, num_feature_);
      vdot = -Dot(dir, dir, num_feature_);
    }
    // shift the history record
    mdot.Shift(); hist.Shift();
    // next n
    if (n < m) n += 1;
    hist.set_num_useful(n);
    // copy gradient to hist[m + n - 1]
    memcpy(hist[m + n - 1], gsub, nsub * sizeof(DType));
    return vdot;
  }
  // line search for given direction
  // return whether there is a descent
  virtual bool BacktrackLineSearch(DType *new_weight,
                                   const DType *dir,
                                   const DType *weight,
                                   double dot_dir_l1grad) {
    utils::Assert(dot_dir_l1grad < 0.0f, "gradient error");
    double alpha = 1.0;
    double backoff = linesearch_backoff;
    // unit descent direction in first iter
    if (hist.num_useful() == 1) {
      alpha = 1.0f / std::sqrt(-dot_dir_l1grad);
      linesearch_backoff = 0.1f;
    }
    double c1 = 1e-4;
    double old_val = this->Eval(weight);
    for (int iter = 0; true; ++iter) {
      if (iter >= max_linesearch_iter) return false;
      AddScale(new_weight, weight, dir, alpha, num_feature_);
      this->FixWeightL1Sign(new_weight, weight, num_feature_);
      double new_val = this->Eval(new_weight);
      if (new_val - old_val <= c1 * dot_dir_l1grad * alpha) break;
      alpha *= backoff;
    }
    return true;
  }
  inline void SetL1Dir(DType *dst,
                        const DType *grad,
                        const DType *weight,
                        size_t size) {
    if (reg_L1 == 0.0) {
      for (size_t i = 0; i < size; ++i) {
        dst[i] = -grad[i];
      }
    } else{
      for (size_t i = 0; i < size; ++i) {
        if (weight[i] > 0.0f) {
          dst[i] = -grad[i] - reg_L1;
        } else if (weight[i] < 0.0f) {
          dst[i] = -grad[i] + reg_L1;
        } else {
          if (grad[i] < -reg_L1) {
            dst[i] = -grad[i] - reg_L1;
          } else if (grad[i] > reg_L1) {
            dst[i] = -grad[i] + reg_L1;
          } else {
            dst[i] = 0.0;
          }
        }
      }
    }
  }
  // fix direction sign to be consistent with proposal
  inline void FixDirL1Sign(DType *dir,
                           const DType *steepdir,
                           size_t size) {
    if (reg_L1 > 0.0) {
      for (size_t i = 0; i < size; ++i) {
        if (dir[i] * steepdir[i] <= 0.0f) {
          dir[i] = 0.0f;
        }
      }
    }
  }
  // fix direction sign to be consistent with proposal
  inline void FixWeightL1Sign(DType *new_weight,
                              const DType *weight,
                              size_t size) {
    if (reg_L1 > 0.0) {
      for (size_t i = 0; i < size; ++i) {
        if (new_weight[i] * weight[i] < 0.0f) {
          new_weight[i] = 0.0f;
        }
      }
    }
  }
  virtual double Eval(const DType *weight) {
    return 0.0f;
  }
 private:
  // dst = lhs + rhs * scale
  inline static void AddScale(DType *dst,
                              const DType *lhs,
                              const DType *rhs,
                              DType scale,
                              size_t size) {
    for (size_t i = 0; i < size; ++i) {
      dst[i] = lhs[i] + rhs[i] * scale;
    }
  }
  // dst = lhs + rhs
  inline static void Minus(DType *dst,
                           const DType *lhs,
                           const DType *rhs,
                           size_t size) {
    for (size_t i = 0; i < size; ++i) {
      dst[i] = lhs[i] - rhs[i];
    }
  }
  inline static double Dot(const DType *lhs,
                           const DType *rhs,
                           size_t size) {
    double res = 0.0;
    for (size_t i = 0; i < size; ++i) {
      res += lhs[i] * rhs[i];
    }
    return res;
  }

  // map rolling array index
  inline static size_t MapIndex(size_t i, size_t offset, size_t size_memory) {
    if (i == 2 * size_memory) return i;
    if (i < size_memory) {
      return (i + offset) % size_memory;
    } else {
      utils::Assert(i < 2 * size_memory,
                    "MapIndex: index exceed bound");
      return (i + offset) % size_memory + size_memory;
    }
  }
  // temp matrix to store the dot product
  struct DotMatrix : public rabit::ISerializable {
   public:
    // intilize the space of rolling array
    inline void Init(size_t size_memory) {
      size_memory_ = size_memory;
      size_t n = size_memory_ * 2 + 1;
      data.resize(n * n, 0.0);
    }
    inline double &Get(size_t i, size_t j)  {
      if (i > j) std::swap(i, j);
      return data[MapIndex(i, offset_, size_memory_) * (size_memory_ * 2 + 1) +
                  MapIndex(j, offset_, size_memory_)];
    }
    // load the shift array
    virtual void Load(rabit::IStream &fi) {
      fi.Read(&size_memory_, sizeof(size_memory_));
      fi.Read(&offset_, sizeof(offset_));
      fi.Read(&data);
    }
    // save the shift array
    virtual void Save(rabit::IStream &fo) const {
      fo.Write(&size_memory_, sizeof(size_memory_));
      fo.Write(&offset_, sizeof(offset_));
      fo.Write(data);
    }
    inline void Shift(void) {
      offset_ = (offset_ + 1) % size_memory_;
    }
    
   private:
    // memory size of L-BFGS
    size_t size_memory_;
    // rolling offset in the current memory
    size_t offset_;
    std::vector<double> data;
  };
  /*! \brief rolling array that carries history information */
  struct HistoryArray : public rabit::ISerializable {
   public:
    HistoryArray(void) : dptr_(NULL) {}
    ~HistoryArray(void) {
      if (dptr_ != NULL) delete [] dptr_;
    }
    // intilize the space of rolling array
    inline void Init(size_t num_col, size_t size_memory) {
      if (dptr_ != NULL &&
          (num_col_ != num_col || size_memory_ != size_memory)) {
        delete dptr_;
      }
      num_col_ = num_col;
      size_memory_ = size_memory;
      stride_ = num_col_;
      offset_ = 0;
      dptr_ = new DType[num_col_ * stride_];
    }
    // fetch element from rolling array
    inline const DType *operator[](size_t i) const {
      return dptr_ + MapIndex(i, offset_, size_memory_) * stride_;
    }
    inline DType *operator[](size_t i) {
      return dptr_ + MapIndex(i, offset_, size_memory_) * stride_;
    }
    // shift array: arr_old -> arr_new
    // for i in [0, size_memory - 1), arr_new[i] = arr_old[i + 1]
    // for i in [size_memory, 2 * size_memory - 1), arr_new[i] = arr_old[i + 1]
    // arr_old[0] and arr_arr[size_memory] will be discarded
    inline void Shift(void) {
      offset_ = (offset_ + 1) % size_memory_;
    }
    inline double CalcDot(size_t i, size_t j) const {
      return Dot((*this)[i], (*this)[j], num_col_);
    }
    // set number of useful memory
    inline const size_t &num_useful(void) const {
      return num_useful_;
    }
    // set number of useful memory
    inline void set_num_useful(size_t num_useful) {
      utils::Assert(num_useful < size_memory_,
                    "num_useful exceed bound");
      num_useful_ = num_useful;
    }
    // load the shift array
    virtual void Load(rabit::IStream &fi) {
      fi.Read(this, sizeof(size_t) * 4);
      this->Init(num_col_, size_memory_);
      for (size_t i = 0; i < num_useful_; ++i) {
        fi.Read((*this)[i], num_col_ * sizeof(DType));
        fi.Read((*this)[i + size_memory_], num_col_ * sizeof(DType));
      }
    }
    // save the shift array
    virtual void Save(rabit::IStream &fi) const {
      fi.Write(this, sizeof(size_t) * 4);
      for (size_t i = 0; i < num_useful_; ++i) {
        fi.Write((*this)[i], num_col_ * sizeof(DType));
        fi.Write((*this)[i + size_memory_], num_col_ * sizeof(DType));
      }
    }
    
   private:
    // number of columns in each of array
    size_t num_col_;
    // stride for each of column for alignment
    size_t stride_;
    // memory size of L-BFGS
    size_t size_memory_;
    // number of useful memory that will be used
    size_t num_useful_;
    // rolling offset in the current memory
    size_t offset_;
    // data pointer
    DType *dptr_;
  };  
  // data structure for LBFGS
  DotMatrix mdot;
  HistoryArray hist;
  size_t num_feature_;
  size_t size_memory_;
  size_t range_begin_;
  size_t range_end_;
  double old_fval;
  // L1 regularization co-efficient
  float reg_L1;
  float linesearch_backoff;
  int max_linesearch_iter;
};
}  // namespace solver
}  // namespace rabit
#endif // RABIT_LBFGS_H_
