/*!
 * Copyright 2019 XGBoost contributors
 */
#ifndef XGBOOST_TREE_CONSTRAINTS_H_
#define XGBOOST_TREE_CONSTRAINTS_H_

#include <dmlc/json.h>
#include <xgboost/logging.h>

#include <cinttypes>
#include <iterator>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <set>

#include "param.h"
#include "../common/span.h"
#include "../common/device_helpers.cuh"

#include <bitset>

namespace xgboost {

__forceinline__ __device__ unsigned long long AtomicOr(unsigned long long* address,
                                                       unsigned long long val) {
  unsigned long long int old = *address, assumed;  // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address, assumed, val | assumed);
  } while (assumed != old);

  return old;
}

__forceinline__ __device__ unsigned long long AtomicAnd(unsigned long long* address,
                                                        unsigned long long val) {
  unsigned long long int old = *address, assumed;  // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address, assumed, val & assumed);
  } while (assumed != old);

  return old;
}

/*!
 * \brief A non-owning type with auxiliary methods defined for manipulating bits.
 */
struct BitField {
  using value_type = uint64_t;

  static value_type constexpr kValueSize = sizeof(value_type) * 8;
  static value_type constexpr kOne = 1UL;  // force uint64_t
  static_assert(kValueSize == 64, "uint64_t should be of 64 bits.");

  struct Pos {
    value_type int_pos {0};
    value_type bit_pos {0};
  };

  common::Span<value_type> bits_;

 public:
  BitField() = default;
  XGBOOST_DEVICE BitField(common::Span<value_type> bits) : bits_{bits} {}
  XGBOOST_DEVICE BitField(BitField const& other) : bits_{other.bits_} {}

  static size_t ComputeStorageSize(size_t size) {
    auto pos = ToBitPos(size);
    if (size < kValueSize) {
      return 1;
    }

    if (pos.bit_pos != 0) {
      return pos.int_pos + 2;
    } else {
      return pos.int_pos + 1;
    }
  }
  XGBOOST_DEVICE static Pos ToBitPos(value_type pos) {
    Pos pos_v;
    if (pos == 0) {
      return pos_v;
    }
    pos_v.int_pos =  pos / kValueSize;
    pos_v.bit_pos =  pos % kValueSize;
    return pos_v;
  }

  __device__ BitField& operator|=(BitField const& rhs) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t min_size = min(bits_.size(), rhs.bits_.size());
    if (tid < min_size) {
      bits_[tid] |= rhs.bits_[tid];
    }
    return *this;
  }
  __device__ BitField& operator&=(BitField const& rhs) {
    size_t min_size = min(bits_.size(), rhs.bits_.size());
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < min_size) {
      bits_[tid] &= rhs.bits_[tid];
    }
    return *this;
  }

  XGBOOST_DEVICE size_t Size() const { return kValueSize * bits_.size(); }

  __device__ void Set(value_type pos) {
    Pos pos_v = ToBitPos(pos);
    value_type& value = bits_[pos_v.int_pos];
    value_type set_bit = kOne << (kValueSize - pos_v.bit_pos - kOne);
    static_assert(sizeof(unsigned long long int) == sizeof(value_type), "");
    AtomicOr(reinterpret_cast<unsigned long long*>(&value), set_bit);
  }
  __device__ void Clear(value_type pos) {
    Pos pos_v = ToBitPos(pos);
    value_type& value = bits_[pos_v.int_pos];
    value_type clear_bit = ~(kOne << (kValueSize - pos_v.bit_pos - kOne));
    static_assert(sizeof(unsigned long long int) == sizeof(value_type), "");
    AtomicAnd(reinterpret_cast<unsigned long long*>(&value), clear_bit);
  }

  XGBOOST_DEVICE bool Check(Pos pos_v) const {
    value_type value = bits_[pos_v.int_pos];
    value_type const test_bit = kOne << (kValueSize - pos_v.bit_pos - kOne);
    value_type result = test_bit & value;
    return static_cast<bool>(result);
  }
  XGBOOST_DEVICE bool Check(value_type pos) const {
    Pos pos_v = ToBitPos(pos);
    return Check(pos_v);
  }

  friend std::ostream& operator<<(std::ostream& os, BitField field) {
    os << "Bits " << "storage size: " << field.bits_.size() << "\n";
    for (size_t i = 0; i < field.bits_.size(); ++i) {
      std::bitset<BitField::kValueSize> set(field.bits_[i]);
      os << set << "\n";
    }
    return os;
  }
};

inline void PrintDeviceBits(std::string name, BitField field) {
  std::cout << "Bits: " << name << std::endl;
  std::vector<BitField::value_type> h_field_bits(field.bits_.size());
  thrust::copy(thrust::device_ptr<BitField::value_type>(field.bits_.data()),
               thrust::device_ptr<BitField::value_type>(field.bits_.data() + field.bits_.size()),
               h_field_bits.data());
  BitField h_field;
  h_field.bits_ = {h_field_bits.data(), h_field_bits.data() + h_field_bits.size()};
  std::cout << h_field;
}

inline void PrintSpan(std::string name, common::Span<int32_t> list) {
  std::cout << name << std::endl;
  std::vector<int32_t> h_list(list.size());
  thrust::copy(thrust::device_ptr<int32_t>(list.data()),
               thrust::device_ptr<int32_t>(list.data() + list.size()),
               h_list.data());
  for (auto v : h_list) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;
}

// Feature interaction constraints built for GPU Hist updater.
struct FeatureInteractionConstraint {
 protected:
  // Whether interaction constraint is used.
  bool has_constraint_;
  // Interaction constraints parsed from string parameter.
  std::vector<std::vector<int32_t>> h_feature_constraints_;

  // The feature interaction constraints as CSR.
  dh::device_vector<int32_t> d_fconstraints_;
  common::Span<int32_t> s_fconstraints_;
  dh::device_vector<int32_t> d_fconstraints_ptr_;
  common::Span<int32_t> s_fconstraints_ptr_;
  // Interaction sets for each feature as CSR.
  dh::device_vector<int32_t> d_sets_;
  dh::device_vector<int32_t> d_sets_ptr_;
  common::Span<int32_t> s_sets_;
  common::Span<int32_t> s_sets_ptr_;
  // Combined features from all interaction sets one feature belongs to.
  dh::device_vector<BitField::value_type> d_feature_buffer_storage_;
  BitField feature_buffer_;

  // Allowed features attached to each node, have n_nodes bitfields,
  // each of size n_features.
  std::vector<dh::device_vector<BitField::value_type>> node_constraints_storage_;
  std::vector<BitField> node_constraints_;
  common::Span<BitField> s_node_constraints_;

  // buffer storing return feature list from Query, of size n_features.
  dh::device_vector<int32_t> result_buffer_;
  common::Span<int32_t> s_result_buffer_;

  // Temp buffers, one bit for each possible feature.
  dh::device_vector<BitField::value_type> output_buffer_bits_storage_;
  BitField output_buffer_bits_;
  dh::device_vector<BitField::value_type> input_buffer_bits_storage_;
  BitField input_buffer_bits_;
  // Clear out all temp buffers
  void ClearBuffers();

 public:
  // size_t Features() { return d_feature_constraints_.size(); }
  size_t Features() const;
  FeatureInteractionConstraint() = default;
  void Configure(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraint(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraint(FeatureInteractionConstraint const& that) = default;
  FeatureInteractionConstraint(FeatureInteractionConstraint&& that) = default;
  /*! \brief Reset before constructing a new tree. */
  void Reset();
  /*! \brief Return a list of features given node id */
  common::Span<int32_t> QueryNode(int32_t nid);
  /*!
   * \brief Return a list of selected features from given feature_list and node id.
   *
   * \param feature_list A list of features
   * \param nid node id
   *
   * \return A list of features picked from `feature_list' that conform to constraints in
   * node.
   */
  common::Span<int32_t> Query(common::Span<int32_t> feature_list, int32_t nid);
  /*! \brief Apply split for node_id. */
  void Split(int32_t node_id, int32_t feature_id, int32_t left_id, int32_t right_id);
};

}      // namespace xgboost
#endif  // XGBOOST_TREE_CONSTRAINTS_H_
