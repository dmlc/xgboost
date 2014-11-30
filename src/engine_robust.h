/*!
 * \file engine_robust.h
 * \brief Robust implementation of AllReduce
 *   using TCP non-block socket and tree-shape reduction.
 *
 *   This implementation considers the failure of nodes
 *   
 * \author Tianqi, Nacho, Tianyi
 */
#ifndef ALLREDUCE_ENGINE_ROBUST_H
#define ALLREDUCE_ENGINE_ROBUST_H
#include "./engine.h"
#include "./engine_base.h"

namespace engine {
/*! \brief implementation of fault tolerant all reduce engine */
class AllReduceRobust : public AllReduceBase {
 public:  
  virtual ~AllReduceRobust(void) {}
  /*!
   * \brief perform in-place allreduce, on sendrecvbuf 
   *        this function is NOT thread-safe
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param type_nbytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   */  
  virtual void AllReduce(void *sendrecvbuf_,
                         size_t type_nbytes,
                         size_t count,           
                         ReduceFunction reducer);
  /*!
   * \brief broadcast data from root to all nodes
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param size the size of the data to be broadcasted
   * \param root the root worker id to broadcast the data
   */
  virtual void Broadcast(void *sendrecvbuf_, size_t total_size, int root);
  /*! 
   * \brief load latest check point
   * \param p_model pointer to the model
   * \return true if there was stored checkpoint and load was successful
   *   false if there was no stored checkpoint, means we are start over gain
   */  
  virtual bool LoadCheckPoint(utils::ISerializable *p_model);
  /*!
   * \brief checkpoint the model, meaning we finished a stage of execution
   * \param p_model pointer to the model
   */
  virtual void CheckPoint(const utils::ISerializable &model);  

 private:
  // constant one byte out of band message to indicate error happening
  // and mark for channel cleanup
  const static char kOOBReset = 95;
  // and mark for channel cleanup, after OOB signal
  const static char kResetMark = 97;
  // and mark for channel cleanup
  const static char kResetAck = 97;
  /*!
   * \brief summary of actions proposed in all nodes
   *  this data structure is used to make consensus decision
   *  about next action to take in the recovery mode
   */
  struct ActionSummary {
    // maximumly allowed sequence id
    const static int kMaxSeq = 1 << 26;
    //---------------------------------------------
    // The following are bit mask of flag used in 
    //----------------------------------------------
    // some node want to load check point
    const static int kLoadCheck = 1;
    // some node want to do check point
    const static int kCheckPoint = 2;
    // check point Ack, we use a two phase message in check point,
    // this is the second phase of check pointing
    const static int kCheckAck = 4;
    // there are difference sequence number the nodes proposed
    // this means we want to do recover execution of the lower sequence
    // action instead of normal execution
    const static int kDiffSeq = 8;
    // constructor
    ActionSummary(void) {}
    // constructor of action 
    ActionSummary(int flag, int minseqno = kMaxSeq) {
      seqcode = (minseqno << 4) | flag;
    }
    // minimum number of all operations
    inline int min_seqno(void) const {
      return seqcode >> 4;
    }
    // whether the operation set contains a load_check
    inline bool load_check(void) const {
      return (seqcode & kLoadCheck) != 0;
    }
    // whether the operation set contains a check point
    inline bool check_point(void) const {
      return (seqcode & kCheckPoint) != 0;
    }
    // whether the operation set contains a check ack
    inline bool check_ack(void) const {
      return (seqcode & kCheckAck) != 0;
    }
    // whether the operation set contains different sequence number
    inline bool diff_seq(void) const {
      return (seqcode & kDiffSeq) != 0;
    }
    // returns the operation flag of the result
    inline int flag(void) const {
      return seqcode & 15;
    }
    // reducer for AllReduce, used to get the result ActionSummary from all nodes
    inline static void Reducer(const void *src_, void *dst_, int len, const MPI::Datatype &dtype) {
      const ActionSummary *src = (const ActionSummary*)src_;
      ActionSummary *dst = (ActionSummary*)dst_;
      for (int i = 0; i < len; ++i) {
        int src_seqno = src[i].min_seqno();
        int dst_seqno = dst[i].min_seqno();
        int flag = src[i].flag() | dst[i].flag();
        if (src_seqno == dst_seqno) {
          dst[i] = ActionSummary(flag, src_seqno);
        } else {
          dst[i] = ActionSummary(flag | kDiffSeq, std::min(src_seqno, dst_seqno));
        }
      }
    }

   private:
    // internel sequence code
    int seqcode;
  };
  /*! \brief data structure to remember result of Bcast and AllReduce calls */
  class ResultBuffer {
   public:
    // constructor
    ResultBuffer(void) {
      this->Clear();
    }
    // clear the existing record
    inline void Clear(void) {
      seqno_.clear(); size_.clear();
      rptr_.clear(); rptr_.push_back(0);
      data_.clear();
    }
    // allocate temporal space for 
    inline void *AllocTemp(size_t type_nbytes, size_t count) {
      size_t size = type_nbytes * count;
      size_t nhop = (size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
      utils::Assert(nhop != 0, "cannot allocate 0 size memory");
      data_.resize(rptr_.back() + nhop);
      return BeginPtr(data_) + rptr_.back();
    }
    // push the result in temp to the 
    inline void PushTemp(int seqid, size_t type_nbytes, size_t count) {
      size_t size = type_nbytes * count;
      size_t nhop = (size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
      if (seqno_.size() != 0) {
        utils::Assert(seqno_.back() < seqid, "PushTemp seqid inconsistent");
      }
      seqno_.push_back(seqid);
      rptr_.push_back(rptr_.back() + nhop);
      size_.push_back(size);
      utils::Assert(data_.size() == rptr_.back(), "PushTemp inconsistent");
    }
    // return the stored result of seqid, if any 
    inline void* Query(int seqid, size_t *p_size) {
      size_t idx = std::lower_bound(seqno_.begin(), seqno_.end(), seqid) - seqno_.begin();
      if (idx == seqno_.size() || seqno_[idx] != seqid) return NULL;
      *p_size = size_[idx];
      return BeginPtr(data_) + rptr_[idx];
    }
   private:
    // sequence number of each 
    std::vector<int> seqno_;
    // pointer to the positions
    std::vector<size_t> rptr_;
    // actual size of each buffer
    std::vector<size_t> size_;
    // content of the buffer
    std::vector<uint64_t> data_;    
  };
  /*!
   * \brief reset the all the existing links by sending Out-of-Band message marker
   *  after this function finishes, all the messages received and sent before in all live links are discarded,
   *  This allows us to get a fresh start after error has happened
   *
   * \return this function can return kSuccess or kSockError
   *         when kSockError is returned, it simply means there are bad sockets in the links,
   *         and some link recovery proceduer is needed
   */
  ReturnType TryResetLinks(void);  
  /*!
   * \brief try to reconnect the broken links
   * \return this function can kSuccess or kSockError
   */
  ReturnType TryReConnectLinks(void);
  /*!
   * \brief if err_type indicates an error
   *         recover links according to the error type reported
   *        if there is no error, return true
   * \param err_type the type of error happening in the system
   * \return true if err_type is kSuccess, false otherwise 
   */
  bool CheckAndRecover(ReturnType err_type);
  /*!
   * \brief try to run recover execution for a request action described by flag and seqno,
   *        the function will keep blocking to run possible recovery operations before the specified action,
   *        until the requested result is received by a recovering procedure,
   *        or the function discovers that the requested action is not yet executed, and return false
   *
   * \param buf the buffer to store the result
   * \param size the total size of the buffer
   * \param flag flag information about the action \sa ActionSummary
   * \param seqno sequence number of the action, if it is special action with flag set, seqno needs to be set to ActionSummary::kMaxSeq
   *
   * \return if this function can return true or false 
 *    - true means buf already set to the
 *           result by recovering procedure, the action is complete, no further action is needed
   *    - false means this is the lastest action that has not yet been executed, need to execute the action
   */
  bool RecoverExec(void *buf, size_t size, int flag, int seqno = ActionSummary::kMaxSeq);  
  /*!
   * \brief try to load check point
   *        
   *        This is a collaborative function called by all nodes
   *        only the nodes with requester set to true really needs to load the check point
   *        other nodes acts as collaborative roles to complete this request
   *
   * \param requester whether current node is the requester
   * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
   * \sa ReturnType
   */
  ReturnType TryLoadCheckPoint(bool requester);
  /*!
   * \brief try to get the result of operation specified by seqno
   *
   *        This is a collaborative function called by all nodes
   *        only the nodes with requester set to true really needs to get the result
   *        other nodes acts as collaborative roles to complete this request
   *
   * \param buf the buffer to store the result, this parameter is only use when current node is requester
   * \param size the total size of the buffer, this parameter is only use when current node is requester
   * \param seqno sequence number of the operation, this is unique index of a operation in current iteration
   * \param requester whether current node is the requester
   * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
   * \sa ReturnType
   */
  ReturnType TryGetResult(void *buf, size_t size, int seqno, bool requester);
  //---- recovery data structure ----
  // call sequence counter, records how many calls we made so far
  // from last call to CheckPoint, LoadCheckPoint
  int seq_counter;
  // result buffer
  ResultBuffer resbuf;
};
}  // namespace engine
#endif // ALLREDUCE_ENGINE_ROBUST_H
