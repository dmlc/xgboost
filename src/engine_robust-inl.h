/*!
 * \file engine_robust-inl.h
 * \brief implementation of inline template function in AllReduceRobust
 *   
 * \author Tianqi, Nacho, Tianyi
 */
#ifndef ALLREDUCE_ENGINE_ROBUST_INL_H
#define ALLREDUCE_ENGINE_ROBUST_INL_H

namespace engine {
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
inline AllReduceRobust::ReturnType
AllReduceRobust::MsgPassing(const NodeType &node_value,
                            std::vector<EdgeType> *p_edge_in,
                            std::vector<EdgeType> *p_edge_out,
                            EdgeType (*func) (const NodeType &node_value,
                                              const std::vector<EdgeType> &edge_in,
                                              size_t out_index)
                            ) {
  return kSuccess;
}
}  // namespace engine
#endif  // ALLREDUCE_ENGINE_ROBUST_INL_H
