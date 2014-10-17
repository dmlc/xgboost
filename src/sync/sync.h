#ifndef XGBOOST_SYNC_SYNC_H_
#define XGBOOST_SYNC_SYNC_H_
/*!
 * \file sync.h
 * \brief interface to do synchronization
 * \author Tianqi Chen
 */
#include <cstdio>
#include <cstring>
#include "../utils/utils.h"
#include <string>

namespace xgboost {
/*! \brief syncrhonizer module that minimumly wraps interface of MPI */
namespace sync {
/*! \brief reduce operator supported */
enum ReduceOp {
  kSum,
  kMax,
  kBitwiseOR
};

/*! \brief get rank of current process */
int GetRank(void);
/*! \brief get total number of process */
int GetWorldSize(void);
/*! \brief get name of processor */
std::string GetProcessorName(void);

/*! 
 * \brief this is used to check if sync module is a true distributed implementation, or simply a dummpy
 */
bool IsDistributed(void);
/*! \brief intiialize the synchronization module */
void Init(int argc, char *argv[]);
/*! \brief finalize syncrhonization module */
void Finalize(void);

/*!
 * \brief in-place all reduce operation 
 * \param sendrecvbuf the in place send-recv buffer
 * \param count count of data
 * \param op reduction function
 */
template<typename DType>
void AllReduce(DType *sendrecvbuf, int count, ReduceOp op);

/*!
 * \brief broadcast an std::string to all others from root
 * \param sendrecv_data the pointer to send or recive buffer,
 *                      receive buffer does not need to be pre-allocated
 *                      and string will be resized to correct length
 * \param root the root of process
 */
void Bcast(std::string *sendrecv_data, int root);

/*! 
 * \brief handle for customized reducer 
 * user do not need to use this, used Reducer instead
 */
class ReduceHandle {
 public:
  // reduce function
  typedef void (ReduceFunction) (const void *src, void *dst, int len);
  // constructor
  ReduceHandle(void);
  // destructor
  ~ReduceHandle(void);
  // initialize the reduce function
  void Init(ReduceFunction redfunc, bool commute = true);
  /*!
   * \brief customized in-place all reduce operation 
   * \param sendrecvbuf the in place send-recv buffer
   * \param n4bytes number of nbytes send through all reduce
   */
  void AllReduce(void *sendrecvbuf, size_t n4bytes);
  
 private:
  // handle data field
  void *handle;
};

// ----- extensions for ease of use ------
/*!
 * \brief template class to make customized reduce and all reduce easy  
 * Do not use reducer directly in the function you call Finalize, because the destructor can happen after Finalize
 * \tparam DType data type that to be reduced
 *   DType must be a struct, with no pointer, and contains a function Reduce(const DType &d);
 */
template<typename DType>
class Reducer {
 public:
  Reducer(void) {
    handle.Init(ReduceInner);
    utils::Assert(sizeof(DType) % sizeof(int) == 0, "struct must be multiple of int");
  }
  /*!
   * \brief customized in-place all reduce operation 
   * \param sendrecvbuf the in place send-recv buffer
   * \param bytes number of 4bytes send through all reduce
   * \param reducer the reducer function
   */
  inline void AllReduce(DType *sendrecvbuf, size_t count) {
    handle.AllReduce(sendrecvbuf, count * kUnit);
  }

 private:
  // unit size 
  static const size_t kUnit = sizeof(DType) / sizeof(int);
  // inner implementation of reducer
  inline static void ReduceInner(const void *src_, void *dst_, int len_) {
    const int *psrc = reinterpret_cast<const int*>(src_);
    int *pdst = reinterpret_cast<int*>(dst_);
    DType tdst, tsrc;
    utils::Assert(len_ % kUnit == 0, "length not divide by size");
    for (size_t i = 0; i < len_; i += kUnit) {
      // use memcpy to avoid alignment issue
      std::memcpy(&tdst, pdst + i, sizeof(tdst));
      std::memcpy(&tsrc, psrc + i, sizeof(tsrc));
      tdst.Reduce(tsrc);
      std::memcpy(pdst + i, &tdst, sizeof(tdst));      
    }
  }
  // function handle
  ReduceHandle handle;
};

}  // namespace sync
}  // namespace xgboost
#endif
