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
#include "../utils/io.h"
#include <string>

namespace MPI {
// forward delcaration of MPI::Datatype, but not include content
class Datatype;
};
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
  typedef void (ReduceFunction) (const void *src, void *dst, int len, const MPI::Datatype &dtype);
  // constructor
  ReduceHandle(void);
  // destructor
  ~ReduceHandle(void);
  /*!
   * \brief initialize the reduce function, with the type the reduce function need to deal with   
   */
  void Init(ReduceFunction redfunc, size_t type_n4bytes, bool commute = true);
  /*!
   * \brief customized in-place all reduce operation 
   * \param sendrecvbuf the in place send-recv buffer
   * \param type_n4bytes unit size of the type, in terms of 4bytes
   * \param count number of elements to send
   */
  void AllReduce(void *sendrecvbuf, size_t type_n4bytes, size_t count);
  /*! \return the number of bytes occupied by the type */
  static int TypeSize(const MPI::Datatype &dtype);
 private:
  // handle data field
  void *handle;
  // handle to the type field
  void *htype;
  // the created type in 4 bytes
  size_t created_type_n4bytes;
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
    handle.Init(ReduceInner, kUnit);
    utils::Assert(sizeof(DType) % sizeof(int) == 0, "struct must be multiple of int");
  }
  /*!
   * \brief customized in-place all reduce operation 
   * \param sendrecvbuf the in place send-recv buffer
   * \param bytes number of 4bytes send through all reduce
   * \param reducer the reducer function
   */
  inline void AllReduce(DType *sendrecvbuf, size_t count) {
    handle.AllReduce(sendrecvbuf, kUnit, count);
  }

 private:
  // unit size 
  static const size_t kUnit = sizeof(DType) / sizeof(int);
  // inner implementation of reducer
  inline static void ReduceInner(const void *src_, void *dst_, int len_, const MPI::Datatype &dtype) {
    const int *psrc = reinterpret_cast<const int*>(src_);
    int *pdst = reinterpret_cast<int*>(dst_);
    DType tdst, tsrc;
    for (size_t i = 0; i < len_; ++i) {
      // use memcpy to avoid alignment issue
      std::memcpy(&tdst, pdst + i * kUnit, sizeof(tdst));
      std::memcpy(&tsrc, psrc + i * kUnit, sizeof(tsrc));
      tdst.Reduce(tsrc);
      std::memcpy(pdst + i * kUnit, &tdst, sizeof(tdst));      
    }
  }
  // function handle
  ReduceHandle handle;
};

/*!
 * \brief template class to make customized reduce, complex reducer handles all the data structure that can be 
 *        serialized/deserialzed into fixed size buffer
 * Do not use reducer directly in the function you call Finalize, because the destructor can happen after Finalize
 * 
 * \tparam DType data type that to be reduced, DType must contain following functions:
 *   (1) Save(IStream &fs)  (2) Load(IStream &fs) (3) Reduce(const DType &d);
 */
template<typename DType>
class SerializeReducer {
 public:
  SerializeReducer(void) {
    handle.Init(ReduceInner, 0);
  }
  /*!
   * \brief customized in-place all reduce operation
   * \param sendrecvobj pointer to the object to be reduced
   * \param max_n4byte maximum amount of memory needed in 4byte
   * \param reducer the reducer function
   */
  inline void AllReduce(DType *sendrecvobj, size_t max_n4byte, size_t count) {
    buffer.resize(max_n4byte * count);
    for (size_t i = 0; i < count; ++i) {
      utils::MemoryFixSizeBuffer fs(BeginPtr(buffer) + i * max_n4byte, max_n4byte * 4);
      sendrecvobj[i].Save(fs);
    }
    handle.AllReduce(BeginPtr(buffer), max_n4byte, count);
    for (size_t i = 0; i < count; ++i) {
      utils::MemoryFixSizeBuffer fs(BeginPtr(buffer) + i * max_n4byte, max_n4byte * 4);
      sendrecvobj[i].Load(fs);
    }
  }

 private:
  // unit size
  // inner implementation of reducer
  inline static void ReduceInner(const void *src_, void *dst_, int len_, const MPI::Datatype &dtype) {
    int nbytes = ReduceHandle::TypeSize(dtype);
    // temp space
    DType tsrc, tdst;
    for (int i = 0; i < len_; ++i) {
      utils::MemoryFixSizeBuffer fsrc((char*)(src_) + i * nbytes, nbytes);
      utils::MemoryFixSizeBuffer fdst((char*)(dst_) + i * nbytes, nbytes);
      tsrc.Load(fsrc);
      tdst.Load(fdst);
      // govern const check
      tdst.Reduce(static_cast<const DType &>(tsrc), nbytes);
      fdst.Seek(0);
      tdst.Save(fdst);
    }
  }
  // function handle
  ReduceHandle handle;
  // reduce buffer
  std::vector<int> buffer;
};

}  // namespace sync
}  // namespace xgboost
#endif
