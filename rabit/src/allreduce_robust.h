/*!
 *  Copyright (c) 2014 by Contributors
 * \file allreduce_robust.h
 * \brief Robust implementation of Allreduce
 *   using TCP non-block socket and tree-shape reduction.
 *
 *   This implementation considers the failure of nodes
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#ifndef RABIT_ALLREDUCE_ROBUST_H_
#define RABIT_ALLREDUCE_ROBUST_H_
#include <vector>
#include <string>
#include <algorithm>
#include "../include/rabit/internal/engine.h"
#include "./allreduce_base.h"

namespace rabit {
namespace engine {
/*! \brief implementation of fault tolerant all reduce engine */
class AllreduceRobust : public AllreduceBase {
 public:
  AllreduceRobust(void);
  virtual ~AllreduceRobust(void) {}
  // initialize the manager
  virtual void Init(int argc, char* argv[]);
  /*! \brief shutdown the engine */
  virtual void Shutdown(void);
  /*!
   * \brief set parameters to the engine
   * \param name parameter name
   * \param val parameter value
   */
  virtual void SetParam(const char *name, const char *val);
  /*!
   * \brief perform in-place allreduce, on sendrecvbuf
   *        this function is NOT thread-safe
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param type_nbytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   * \param prepare_func Lazy preprocessing function, lazy prepare_fun(prepare_arg)
   *                     will be called by the function before performing Allreduce, to intialize the data in sendrecvbuf_.
   *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
   * \param prepare_arg argument used to passed into the lazy preprocessing function
   */
  virtual void Allreduce(void *sendrecvbuf_,
                         size_t type_nbytes,
                         size_t count,
                         ReduceFunction reducer,
                         PreprocFunction prepare_fun = NULL,
                         void *prepare_arg = NULL);
  /*!
   * \brief broadcast data from root to all nodes
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param size the size of the data to be broadcasted
   * \param root the root worker id to broadcast the data
   */
  virtual void Broadcast(void *sendrecvbuf_, size_t total_size, int root);
  /*!
   * \brief load latest check point
   * \param global_model pointer to the globally shared model/state
   *   when calling this function, the caller need to gauranttees that global_model
   *   is the same in all nodes
   * \param local_model pointer to local model, that is specific to current node/rank
   *   this can be NULL when no local model is needed
   *
   * \return the version number of check point loaded
   *     if returned version == 0, this means no model has been CheckPointed
   *     the p_model is not touched, user should do necessary initialization by themselves
   *
   *   Common usage example:
   *      int iter = rabit::LoadCheckPoint(&model);
   *      if (iter == 0) model.InitParameters();
   *      for (i = iter; i < max_iter; ++i) {
   *        do many things, include allreduce
   *        rabit::CheckPoint(model);
   *      }
   *
   * \sa CheckPoint, VersionNumber
   */
  virtual int LoadCheckPoint(Serializable *global_model,
                             Serializable *local_model = NULL);
  /*!
   * \brief checkpoint the model, meaning we finished a stage of execution
   *  every time we call check point, there is a version number which will increase by one
   *
   * \param global_model pointer to the globally shared model/state
   *   when calling this function, the caller need to gauranttees that global_model
   *   is the same in all nodes
   * \param local_model pointer to local model, that is specific to current node/rank
   *   this can be NULL when no local state is needed
   *
   * NOTE: local_model requires explicit replication of the model for fault-tolerance, which will
   *       bring replication cost in CheckPoint function. global_model do not need explicit replication.
   *       So only CheckPoint with global_model if possible
   *
   * \sa LoadCheckPoint, VersionNumber
   */
  virtual void CheckPoint(const Serializable *global_model,
                          const Serializable *local_model = NULL) {
    this->CheckPoint_(global_model, local_model, false);
  }
  /*!
   * \brief This function can be used to replace CheckPoint for global_model only,
   *   when certain condition is met(see detailed expplaination).
   *
   *   This is a "lazy" checkpoint such that only the pointer to global_model is
   *   remembered and no memory copy is taken. To use this function, the user MUST ensure that:
   *   The global_model must remain unchanged util last call of Allreduce/Broadcast in current version finishs.
   *   In another words, global_model model can be changed only between last call of
   *   Allreduce/Broadcast and LazyCheckPoint in current version
   *
   *   For example, suppose the calling sequence is:
   *   LazyCheckPoint, code1, Allreduce, code2, Broadcast, code3, LazyCheckPoint
   *
   *   If user can only changes global_model in code3, then LazyCheckPoint can be used to
   *   improve efficiency of the program.
   * \param global_model pointer to the globally shared model/state
   *   when calling this function, the caller need to gauranttees that global_model
   *   is the same in all nodes
   * \sa LoadCheckPoint, CheckPoint, VersionNumber
   */
  virtual void LazyCheckPoint(const Serializable *global_model) {
    this->CheckPoint_(global_model, NULL, true);
  }
  /*!
   * \brief explicitly re-init everything before calling LoadCheckPoint
   *    call this function when IEngine throw an exception out,
   *    this function is only used for test purpose
   */
  virtual void InitAfterException(void) {
    // simple way, shutdown all links
    for (size_t i = 0; i < all_links.size(); ++i) {
      if (!all_links[i].sock.BadSocket()) all_links[i].sock.Close();
    }
    ReConnectLinks("recover");
  }

 protected:
  // constant one byte out of band message to indicate error happening
  // and mark for channel cleanup
  static const char kOOBReset = 95;
  // and mark for channel cleanup, after OOB signal
  static const char kResetMark = 97;
  // and mark for channel cleanup
  static const char kResetAck = 97;
  /*! \brief type of roles each node can play during recovery */
  enum RecoverType {
    /*! \brief current node have data */
    kHaveData = 0,
    /*! \brief current node request data */
    kRequestData = 1,
    /*! \brief current node only helps to pass data around */
    kPassData = 2
  };
  /*!
   * \brief summary of actions proposed in all nodes
   *  this data structure is used to make consensus decision
   *  about next action to take in the recovery mode
   */
  struct ActionSummary {
    // maximumly allowed sequence id
    static const int kSpecialOp = (1 << 26);
    // special sequence number for local state checkpoint
    static const int kLocalCheckPoint = (1 << 26) - 2;
    // special sequnce number for local state checkpoint ack signal
    static const int kLocalCheckAck = (1 << 26) - 1;
    //---------------------------------------------
    // The following are bit mask of flag used in
    //----------------------------------------------
    // some node want to load check point
    static const int kLoadCheck = 1;
    // some node want to do check point
    static const int kCheckPoint = 2;
    // check point Ack, we use a two phase message in check point,
    // this is the second phase of check pointing
    static const int kCheckAck = 4;
    // there are difference sequence number the nodes proposed
    // this means we want to do recover execution of the lower sequence
    // action instead of normal execution
    static const int kDiffSeq = 8;
    // constructor
    ActionSummary(void) {}
    // constructor of action
    explicit ActionSummary(int flag, int minseqno = kSpecialOp) {
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
    // reducer for Allreduce, get the result ActionSummary from all nodes
    inline static void Reducer(const void *src_, void *dst_,
                               int len, const MPI::Datatype &dtype) {
      const ActionSummary *src = (const ActionSummary*)src_;
      ActionSummary *dst = reinterpret_cast<ActionSummary*>(dst_);
      for (int i = 0; i < len; ++i) {
        int src_seqno = src[i].min_seqno();
        int dst_seqno = dst[i].min_seqno();
        int flag = src[i].flag() | dst[i].flag();
        if (src_seqno == dst_seqno) {
          dst[i] = ActionSummary(flag, src_seqno);
        } else {
          dst[i] = ActionSummary(flag | kDiffSeq,
                                 std::min(src_seqno, dst_seqno));
        }
      }
    }

   private:
    // internel sequence code
    int seqcode;
  };
  /*! \brief data structure to remember result of Bcast and Allreduce calls */
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
    // allocate temporal space
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
      size_t idx = std::lower_bound(seqno_.begin(),
                                    seqno_.end(), seqid) - seqno_.begin();
      if (idx == seqno_.size() || seqno_[idx] != seqid) return NULL;
      *p_size = size_[idx];
      return BeginPtr(data_) + rptr_[idx];
    }
    // drop last stored result
    inline void DropLast(void) {
      utils::Assert(seqno_.size() != 0, "there is nothing to be dropped");
      seqno_.pop_back();
      rptr_.pop_back();
      size_.pop_back();
      data_.resize(rptr_.back());
    }
    // the sequence number of last stored result
    inline int LastSeqNo(void) const {
      if (seqno_.size() == 0) return -1;
      return seqno_.back();
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
   * \brief internal consistency check function,
   *  use check to ensure user always call CheckPoint/LoadCheckPoint
   *  with or without local but not both, this function will set the approperiate settings
   *  in the first call of LoadCheckPoint/CheckPoint
   *
   * \param with_local whether the user calls CheckPoint with local model
   */
  void LocalModelCheck(bool with_local);
  /*!
   * \brief internal implementation of checkpoint, support both lazy and normal way
   *
   * \param global_model pointer to the globally shared model/state
   *   when calling this function, the caller need to gauranttees that global_model
   *   is the same in all nodes
   * \param local_model pointer to local model, that is specific to current node/rank
   *   this can be NULL when no local state is needed
   * \param lazy_checkpt whether the action is lazy checkpoint
   *
   * \sa CheckPoint, LazyCheckPoint
   */
  void CheckPoint_(const Serializable *global_model,
                   const Serializable *local_model,
                   bool lazy_checkpt);
  /*!
   * \brief reset the all the existing links by sending Out-of-Band message marker
   *  after this function finishes, all the messages received and sent
   *  before in all live links are discarded,
   *  This allows us to get a fresh start after error has happened
   *
   *  TODO(tqchen): this function is not yet functioning was not used by engine,
   *   simple resetlink and reconnect strategy is used
   *
   * \return this function can return kSuccess or kSockError
   *         when kSockError is returned, it simply means there are bad sockets in the links,
   *         and some link recovery proceduer is needed
   */
  ReturnType TryResetLinks(void);
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
   * \param seqno sequence number of the action, if it is special action with flag set,
   *        seqno needs to be set to ActionSummary::kSpecialOp
   *
   * \return if this function can return true or false
   *    - true means buf already set to the
   *           result by recovering procedure, the action is complete, no further action is needed
   *    - false means this is the lastest action that has not yet been executed, need to execute the action
   */
  bool RecoverExec(void *buf, size_t size, int flag,
                   int seqno = ActionSummary::kSpecialOp);
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
   * \param buf the buffer to store the result, this parameter is only used when current node is requester
   * \param size the total size of the buffer, this parameter is only used when current node is requester
   * \param seqno sequence number of the operation, this is unique index of a operation in current iteration
   * \param requester whether current node is the requester
   * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
   * \sa ReturnType
   */
  ReturnType TryGetResult(void *buf, size_t size, int seqno, bool requester);
  /*!
   * \brief try to decide the routing strategy for recovery
   * \param role the current role of the node
   * \param p_size used to store the size of the message, for node in state kHaveData,
   *               this size must be set correctly before calling the function
   *               for others, this surves as output parameter

   * \param p_recvlink used to store the link current node should recv data from, if necessary
   *          this can be -1, which means current node have the data
   * \param p_req_in used to store the resulting vector, indicating which link we should send the data to
   *
   * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
   * \sa ReturnType, TryRecoverData
   */
  ReturnType TryDecideRouting(RecoverType role,
                              size_t *p_size,
                              int *p_recvlink,
                              std::vector<bool> *p_req_in);
  /*!
   * \brief try to finish the data recovery request,
   *        this function is used together with TryDecideRouting
   * \param role the current role of the node
   * \param sendrecvbuf_ the buffer to store the data to be sent/recived
   *          - if the role is kHaveData, this stores the data to be sent
   *          - if the role is kRequestData, this is the buffer to store the result
   *          - if the role is kPassData, this will not be used, and can be NULL
   * \param size the size of the data, obtained from TryDecideRouting
   * \param recv_link the link index to receive data, if necessary, obtained from TryDecideRouting
   * \param req_in the request of each link to send data, obtained from TryDecideRouting
   *
   * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
   * \sa ReturnType, TryDecideRouting
   */
  ReturnType TryRecoverData(RecoverType role,
                            void *sendrecvbuf_,
                            size_t size,
                            int recv_link,
                            const std::vector<bool> &req_in);
  /*!
   * \brief try to recover the local state, making each local state to be the result of itself
   *        plus replication of states in previous num_local_replica hops in the ring
   *
   * The input parameters must contain the valid local states available in current nodes,
   * This function try ist best to "complete" the missing parts of local_rptr and local_chkpt
   * If there is sufficient information in the ring, when the function returns, local_chkpt will
   * contain num_local_replica + 1 checkpoints (including the chkpt of this node)
   * If there is no sufficient information in the ring, this function the number of checkpoints
   * will be less than the specified value
   *
   * \param p_local_rptr the pointer to the segment pointers in the states array
   * \param p_local_chkpt the pointer to the storage of local check points
   * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
   * \sa ReturnType
   */
  ReturnType TryRecoverLocalState(std::vector<size_t> *p_local_rptr,
                                  std::string *p_local_chkpt);
  /*!
   * \brief try to checkpoint local state, this function is called in normal executation phase
   *    of checkpoint that contains local state
o   *  the input state must exactly one saved state(local state of current node),
   *  after complete, this function will get local state from previous num_local_replica nodes and put them
   *  into local_chkpt and local_rptr
   *
   *  It is also OK to call TryRecoverLocalState instead,
   *  TryRecoverLocalState makes less assumption about the input, and requires more communications
   *
   * \param p_local_rptr the pointer to the segment pointers in the states array
   * \param p_local_chkpt the pointer to the storage of local check points
   * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
   * \sa ReturnType, TryRecoverLocalState
   */
  ReturnType TryCheckinLocalState(std::vector<size_t> *p_local_rptr,
                                  std::string *p_local_chkpt);
  /*!
   * \brief perform a ring passing to receive data from prev link, and sent data to next link
   *  this allows data to stream over a ring structure
   *  sendrecvbuf[0:read_ptr] are already provided by current node
   *  current node will recv sendrecvbuf[read_ptr:read_end] from prev link
   *  current node will send sendrecvbuf[write_ptr:write_end] to next link
   *  write_ptr will wait till the data is readed before sending the data
   *  this function requires read_end >= write_end
   *
   * \param sendrecvbuf_ the place to hold the incoming and outgoing data
   * \param read_ptr the initial read pointer
   * \param read_end the ending position to read
   * \param write_ptr the initial write pointer
   * \param write_end the ending position to write
   * \param read_link pointer to link to previous position in ring
   * \param write_link pointer to link of next position in ring
   */
  ReturnType RingPassing(void *senrecvbuf_,
                         size_t read_ptr,
                         size_t read_end,
                         size_t write_ptr,
                         size_t write_end,
                         LinkRecord *read_link,
                         LinkRecord *write_link);
  /*!
   * \brief run message passing algorithm on the allreduce tree
   *        the result is edge message stored in p_edge_in and p_edge_out
   * \param node_value the value associated with current node
   * \param p_edge_in used to store input message from each of the edge
   * \param p_edge_out used to store output message from each of the edge
   * \param func a function that defines the message passing rule
   *        Parameters of func:
   *           - node_value same as node_value in the main function
   *           - edge_in the array of input messages from each edge,
   *                     this includes the output edge, which should be excluded
   *           - out_index array the index of output edge, the function should
   *                       exclude the output edge when compute the message passing value
   *        Return of func:
   *           the function returns the output message based on the input message and node_value
   *
   * \tparam EdgeType type of edge message, must be simple struct
   * \tparam NodeType type of node value
   */
  template<typename NodeType, typename EdgeType>
  inline ReturnType MsgPassing(const NodeType &node_value,
                               std::vector<EdgeType> *p_edge_in,
                               std::vector<EdgeType> *p_edge_out,
                               EdgeType(*func)
                               (const NodeType &node_value,
                                const std::vector<EdgeType> &edge_in,
                                size_t out_index));
  //---- recovery data structure ----
  // the round of result buffer, used to mode the result
  int result_buffer_round;
  // result buffer of all reduce
  ResultBuffer resbuf;
  // last check point global model
  std::string global_checkpoint;
  // lazy checkpoint of global model
  const Serializable *global_lazycheck;
  // number of replica for local state/model
  int num_local_replica;
  // number of default local replica
  int default_local_replica;
  // flag to decide whether local model is used, -1: unknown, 0: no, 1:yes
  int use_local_model;
  // number of replica for global state/model
  int num_global_replica;
  // number of times recovery happens
  int recover_counter;
  // --- recovery data structure for local checkpoint
  // there is two version of the data structure,
  // at one time one version is valid and another is used as temp memory
  // pointer to memory position in the local model
  // local model is stored in CSR format(like a sparse matrices)
  // local_model[rptr[0]:rptr[1]] stores the model of current node
  // local_model[rptr[k]:rptr[k+1]] stores the model of node in previous k hops
  std::vector<size_t> local_rptr[2];
  // storage for local model replicas
  std::string local_chkpt[2];
  // version of local checkpoint can be 1 or 0
  int local_chkpt_version;
};
}  // namespace engine
}  // namespace rabit
// implementation of inline template function
#include "./allreduce_robust-inl.h"
#endif  // RABIT_ALLREDUCE_ROBUST_H_
