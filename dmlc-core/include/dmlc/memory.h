/*!
 *  Copyright (c) 2015 by Contributors
 * \file memory.h
 * \brief Additional memory hanlding utilities.
 */
#ifndef DMLC_MEMORY_H_
#define DMLC_MEMORY_H_

#include <vector>
#include "./base.h"
#include "./logging.h"
#include "./thread_local.h"

namespace dmlc {

/*!
 * \brief A memory pool that allocate memory of fixed size and alignment.
 * \tparam size The size of each piece.
 * \tparam align The alignment requirement of the memory.
 */
template<size_t size, size_t align>
class MemoryPool {
 public:
  /*! \brief constructor */
  MemoryPool() {
    static_assert(align % alignof(LinkedList) == 0,
                  "alignment requirement failed.");
    curr_page_.reset(new Page());
  }
  /*! \brief allocate a new memory of size */
  inline void* allocate() {
    if (head_ != nullptr) {
      LinkedList* ret = head_;
      head_ = head_->next;
      return ret;
    } else {
      if (page_ptr_ < kPageSize) {
        return &(curr_page_->data[page_ptr_++]);
      } else {
        allocated_.push_back(std::move(curr_page_));
        curr_page_.reset(new Page());
        page_ptr_ = 1;
        return &(curr_page_->data[0]);
      }
    }
  }
  /*!
   * \brief deallocate a piece of memory
   * \param p The pointer to the memory to be de-allocated.
   */
  inline void deallocate(void* p) {
    LinkedList* ptr = static_cast<LinkedList*>(p);
    ptr->next = head_;
    head_ = ptr;
  }

 private:
  // page size of each member
  static const int kPageSize = ((1 << 22) / size);
  // page to be requested.
  struct Page {
    typename std::aligned_storage<size, align>::type data[kPageSize];
  };
  // internal linked list structure.
  struct LinkedList {
    LinkedList* next{nullptr};
  };
  // head of free list
  LinkedList* head_{nullptr};
  // current free page
  std::unique_ptr<Page> curr_page_;
  // pointer to the current free page position.
  size_t page_ptr_{0};
  // allocated pages.
  std::vector<std::unique_ptr<Page> > allocated_;
};


/*!
 * \brief A thread local allocator that get memory from a threadlocal memory pool.
 * This is suitable to allocate objects that do not cross thread.
 * \tparam T the type of the data to be allocated.
 */
template<typename T>
class ThreadlocalAllocator {
 public:
  /*! \brief pointer type */
  typedef T* pointer;
  /*! \brief const pointer type */
  typedef const T* const_ptr;
  /*! \brief value type */
  typedef T value_type;
  /*! \brief default constructor */
  ThreadlocalAllocator() {}
  /*!
   * \brief constructor from another allocator
   * \param other another allocator
   * \tparam U another type
   */
  template<typename U>
  ThreadlocalAllocator(const ThreadlocalAllocator<U>& other) {}
  /*!
   * \brief allocate memory
   * \param n number of blocks
   * \return an uninitialized memory of type T.
   */
  inline T* allocate(size_t n) {
    CHECK_EQ(n, 1);
    typedef ThreadLocalStore<MemoryPool<sizeof(T), alignof(T)> > Store;
    return static_cast<T*>(Store::Get()->allocate());
  }
  /*!
   * \brief deallocate memory
   * \param p a memory to be returned.
   * \param n number of blocks
   */
  inline void deallocate(T* p, size_t n) {
    CHECK_EQ(n, 1);
    typedef ThreadLocalStore<MemoryPool<sizeof(T), alignof(T)> > Store;
    Store::Get()->deallocate(p);
  }
};


/*!
 * \brief a shared pointer like type that allocate object
 *   from a threadlocal object pool. This object is not thread-safe
 *   but can be faster than shared_ptr in certain usecases.
 * \tparam T the data type.
 */
template<typename T>
struct ThreadlocalSharedPtr {
 public:
  /*! \brief default constructor */
  ThreadlocalSharedPtr() : block_(nullptr) {}
  /*!
   * \brief constructor from nullptr
   * \param other the nullptr type
   */
  ThreadlocalSharedPtr(std::nullptr_t other) : block_(nullptr) {}  // NOLINT(*)
  /*!
   * \brief copy constructor
   * \param other another pointer.
   */
  ThreadlocalSharedPtr(const ThreadlocalSharedPtr<T>& other)
      : block_(other.block_) {
    IncRef(block_);
  }
  /*!
   * \brief move constructor
   * \param other another pointer.
   */
  ThreadlocalSharedPtr(ThreadlocalSharedPtr<T>&& other)
      : block_(other.block_) {
    other.block_ = nullptr;
  }
  /*!
   * \brief destructor
   */
  ~ThreadlocalSharedPtr() {
    DecRef(block_);
  }
  /*!
   * \brief move assignment
   * \param other another object to be assigned.
   * \return self.
   */
  inline ThreadlocalSharedPtr<T>& operator=(ThreadlocalSharedPtr<T>&& other) {
    DecRef(block_);
    block_ = other.block_;
    other.block_ = nullptr;
    return *this;
  }
  /*!
   * \brief copy assignment
   * \param other another object to be assigned.
   * \return self.
   */
  inline ThreadlocalSharedPtr<T> &operator=(const ThreadlocalSharedPtr<T>& other) {
    DecRef(block_);
    block_ = other.block_;
    IncRef(block_);
    return *this;
  }
  /*! \brief check if nullptr */
  inline bool operator==(std::nullptr_t other) const {
    return block_ == nullptr;
  }
  /*!
   * \return get the pointer content.
   */
  inline T* get() const {
    if (block_ == nullptr) return nullptr;
    return reinterpret_cast<T*>(&(block_->data));
  }
  /*!
   * \brief reset the pointer to nullptr.
   */
  inline void reset() {
    DecRef(block_);
    block_ = nullptr;
  }
  /*! \return if use_count == 1*/
  inline bool unique() const {
    if (block_ == nullptr) return false;
    return block_->use_count_ == 1;
  }
  /*! \return dereference pointer */
  inline T* operator*() const {
    return reinterpret_cast<T*>(&(block_->data));
  }
  /*! \return dereference pointer */
  inline T* operator->() const {
    return reinterpret_cast<T*>(&(block_->data));
  }
  /*!
   * \brief create a new space from threadlocal storage and return it.
   * \tparam Args the arguments.
   * \param args The input argument
   * \return the allocated pointer.
   */
  template <typename... Args>
  inline static ThreadlocalSharedPtr<T> Create(Args&&... args) {
    ThreadlocalAllocator<RefBlock> arena;
    ThreadlocalSharedPtr<T> p;
    p.block_ = arena.allocate(1);
    p.block_->use_count_ = 1;
    new (&(p.block_->data)) T(std::forward<Args>(args)...);
    return p;
  }

 private:
  // internal reference block
  struct RefBlock {
    typename std::aligned_storage<sizeof(T), alignof(T)>::type data;
    unsigned use_count_;
  };
  // decrease ref counter
  inline static void DecRef(RefBlock* block) {
    if (block != nullptr) {
      if (--block->use_count_ == 0) {
        ThreadlocalAllocator<RefBlock> arena;
        T* dptr = reinterpret_cast<T*>(&(block->data));
        dptr->~T();
        arena.deallocate(block, 1);
      }
    }
  }
  // increase ref counter
  inline static void IncRef(RefBlock* block) {
    if (block != nullptr) {
      ++block->use_count_;
    }
  }
  // internal block
  RefBlock *block_;
};

}  // namespace dmlc

#endif  // DMLC_MEMORY_H_
