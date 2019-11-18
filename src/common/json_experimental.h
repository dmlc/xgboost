/*!
 * Copyright 2019 by XGBoost Contributors
 *
 * \brief A JSON parser/generator implementation, no utf-8 support yet.
 */
#ifndef XGBOOST_COMMON_JSON_EXPERIMENTAL_H_
#define XGBOOST_COMMON_JSON_EXPERIMENTAL_H_

#include <cinttypes>
#include <vector>
#include <string>
#include <tuple>
#include <cstring>
#include <limits>
#include <algorithm>
#include <utility>
#include <map>

#include "xgboost/span.h"
#include "xgboost/logging.h"

namespace xgboost {
namespace experimental {

/*\brief Types of JSON value.
 *
 * Inspired by sajson, this type is intentionally packed into 3 bits.
 */
enum class ValueKind : std::uint8_t {
  kTrue = 0x0,
  kFalse = 0x1,
  kInteger = 0x2,
  kNumber = 0x3,
  kString = 0x4,

  kArray = 0x5,
  kObject = 0x6,

  kNull = 0x7
};

inline std::string KindStr(ValueKind kind) {
  switch (kind) {
    case ValueKind::kTrue:
      return "ture";
    case ValueKind::kFalse:
      return "false";
    case ValueKind::kInteger:
      return "integer";
    case ValueKind::kNumber:
      return "number";
    case ValueKind::kString:
      return "string";
    case ValueKind::kArray:
      return "array";
    case ValueKind::kObject:
      return "object";
    case ValueKind::kNull:
      return "null";
  }
  return "";
}

/*! \brief A mutable string_view. */
template <typename CharT>
class StringRefImpl {
 public:
  using pointer = CharT*;
  using iterator = pointer;
  using const_iterator = pointer const;

  using value_type = CharT;
  using traits_type = std::char_traits<CharT>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using reference = CharT&;
  using const_reference = CharT const&;

 private:
  CharT* chars_;
  size_t size_;

 public:
  StringRefImpl() : chars_{nullptr}, size_{0} {}
  StringRefImpl(std::string& str) : chars_{&str[0]}, size_{str.size()} {} // NOLINT
  StringRefImpl(std::string const& str) : chars_{str.data()}, size_{str.size()} {} // NOLINT
  StringRefImpl(CharT* chars, size_t size) : chars_{chars}, size_{size} {}
  StringRefImpl(CharT* chars) : chars_{chars}, size_{traits_type::length(chars)} {}  // NOLINT

  const_iterator cbegin() const { return chars_; }  // NOLINT
  const_iterator cend()   const { return chars_ + size(); }  // NOLINT

  iterator begin() { return chars_; }         // NOLINT
  iterator end() { return chars_ + size(); }  // NOLINT

  pointer data() const { return chars_; }     // NOLINT

  size_t size() const { return size_; };      // NOLINT
  CharT const& operator[](size_t i) const { return chars_[i]; }
  CharT const& at(size_t i) const {  // NOLINT
    CHECK_LT(i, size_); return (*this)[i];
  }

  CharT front() const { CHECK_NE(size_, 0); return chars_[0]; }
  CharT back() const { CHECK_NE(size_, 0); return chars_[size_ - 1]; }

  std::string Copy() const {
    std::string str;
    str.resize(this->size());
    std::memcpy(&str[0], chars_, this->size());
    return str;
  }

  bool operator==(StringRefImpl<CharT> that) const {
    return common::Span<CharT const>{chars_, size_} ==
           common::Span<CharT const> {that.data(), that.size()};
  }
  bool operator!=(StringRefImpl<CharT> const& that) const {
    return !(that == *this);
  }
  bool operator<(StringRefImpl const &that) {
    return common::Span<CharT>{chars_, size_} < common::Span<CharT> {that.data(), that.size()};
  }
  bool operator>(StringRefImpl const &that) {
    return common::Span<CharT>{chars_, size_} > common::Span<CharT> {that.data(), that.size()};
  }
  bool operator<=(StringRefImpl const &that) {
    return common::Span<CharT>{chars_, size_} <= common::Span<CharT> {that.data(), that.size()};
  }
  bool operator>=(StringRefImpl const &that) {
    return common::Span<CharT>{chars_, size_} >= common::Span<CharT> {that.data(), that.size()};
  }

  friend std::ostream& operator<<(std::ostream& os, StringRefImpl str) {
    for (auto v : str) {
      os << v;
    }
    return os;
  }
};

using StringRef = StringRefImpl<std::string::value_type>;
using ConstStringRef = StringRefImpl<std::string::value_type const>;

/*\brief An iterator for looping through Object members.  Inspired by rapidjson. */
template <typename ValueType, bool IsConst>
class ElemIterator {
  typename std::conditional<IsConst, ValueType const*, ValueType *>::type ptr_;
  size_t index_;

 public:
  using reference = typename std::conditional<IsConst, ValueType const, ValueType>::type&;
  using difference_type = std::ptrdiff_t;
  using value_type = typename std::remove_reference<
      typename std::remove_cv<ValueType>::type>::type;

  ElemIterator() = default;
  ElemIterator(ValueType *p, size_t i) : ptr_{p}, index_{i} {}
  template <typename std::enable_if<IsConst>::type* = nullptr>
  ElemIterator(ValueType const *p, size_t i) : ptr_{p}, index_{i} {}
  ElemIterator(ElemIterator const &that) : ptr_{that.ptr_}, index_{that.index_} {}
  // prefix
  ElemIterator& operator++() {
    index_++;
    return *this;
  }
  // postfix
  ElemIterator operator++(int) {
    auto ret = *this;
    ++(*this);
    return ret;
  }

  value_type operator*() const {
    CHECK(ptr_);
    auto v = ptr_->GetMemberByIndex(index_);
    return v;
  }

  ConstStringRef Key() const {
    CHECK(ptr_);
    return ptr_->GetKeyByIndex(index_);
  }

  value_type operator[](difference_type i) const {
    CHECK(ptr_);
    CHECK_LT(i + index_, ptr_->Length());
    ptr_->GetMemberByIndex(i + index_);
  }

  bool operator==(ElemIterator const& that) const {
    return index_ == that.index_ && ptr_ == that.ptr_;
  }
  bool operator!=(ElemIterator const &that) const {
    return !(that == *this);
  }
};

/*! \brief Commom utilitis for handling compact JSON type.  The type implementation is
 *  inspired by sajson. */
struct JsonTypeHandler {
 public:
  // number of bits required to store the type information.
  static size_t constexpr kTypeBits   { 3   };
  // 00...0111, get the first 3 bits where type information is stored.
  static size_t constexpr kTypeMask   { 0x7 };

 public:
  // return type information from typed offset.
  static ValueKind GetType(size_t typed_offset) {
    auto uint_type = static_cast<uint8_t>(typed_offset & kTypeMask);
    return static_cast<ValueKind>(uint_type);
  }
  // return offset information from typed offset.
  static size_t GetOffset(size_t typed_offset) {
    return typed_offset >> kTypeBits;
  }
  // Split up the typed offset into type and offset, combination of above functions.
  static std::pair<ValueKind, size_t> GetTypeOffset(size_t typed_offset) {
    auto uint_type = static_cast<uint8_t>(typed_offset & kTypeMask);
    auto offset = typed_offset >> kTypeBits;
    return {static_cast<ValueKind>(uint_type), offset};
  }
  // Pack type information into a tree pointer.
  static size_t constexpr MakeTypedOffset(size_t ptr, ValueKind kind) {
    return ptr << kTypeBits | static_cast<size_t>(kind);
  }
};

/*\brief A view over std::vector<T>. */
template <typename T>
class StorageView {
  std::vector<T> *storage_ref_;
  void Resize(size_t n) { this->storage_ref_->resize(n); }

 public:
  explicit StorageView(std::vector<T> *storage_ref) : storage_ref_{storage_ref} {}
  size_t Top() const { return storage_ref_->size(); }
  common::Span<T> Access() const {
    return common::Span<T>{storage_ref_->data(), storage_ref_->size()};
  }
  void Expand(size_t n) { this->Resize(storage_ref_->size() + n); }

  template <typename V>
  void Push(V* value, size_t n) {
    size_t old = storage_ref_->size();
    this->Expand(sizeof(V) / sizeof(T) * n);
    std::memcpy(storage_ref_->data() + old, value, n * sizeof(V));
  }
  std::vector<T> *Data() const { return storage_ref_; }
};

class Document;
/*
 * \brief The real implementation for JSON values.
 *
 * Memory layout:
 *
 *   Document class holds all the allocated memory, which includes one block for JSON
 *   tree, and another block for JSON data.  Each JSON value holds a `self_` offset
 *   pointing it's location in the tree memory block.  First 3 bits of tree[self_] packs
 *   the type information of this value, while the rest of the bits is used by value
 *   itself to index its data, depending on the value type.
 *
 *     - Numeric inlcuding integer and float:
 *         `tree[self_]` points to data memory holding the actual integer or float.
 *     - Null, True, False
 *         Only first 3 bits of `tree[self_]` is used, the information is directly encoded
 *         in it's type.
 *     - Array
 *         `tree[self_]` points to a table stored in tree, which stores the indices of
 *         `self_` for each array element.
 *     - Object
 *          `tree[self_]` points to a table stored in tree, which stores the key begin and
 *          end, also the offsets of `self_` for its members.
 *     - String
 *          Points to a struct containing the offsets of begin and end in data memory block.
 *
 *  A value must be "derived" from a document, meaning that you can't simply define a
 *  value (default constructor is implicitly deleted).  A value must be created from an
 *  existing document or another value that is an Object or Array.  This simplifies
 *  memory management.
 */
template <typename Container, size_t kElementEnd = std::numeric_limits<size_t>::max()>
class ValueImpl {
 protected:
  // Using inheritence is not idea as Document would go out of scope before this base
  // class, while memory buffer is held in Document;
  friend Document;
  using ValueImplT = ValueImpl;

  // The document class which contains all the allocated memory.
  Container* handler_;
  // A `this` pointer pointing to JSON tree, with type information written in first 3 bits.
  size_t self_ {0};
  // Storing the type information for this node.
  ValueKind kind_ { ValueKind::kNull };
  bool is_view_ {false};
  bool finalised_ {false};

 public:
  // This is a node in the singly linked list
  struct ObjectElement {
    /*\brief Address of value in JSON tree. */
    size_t value;
    /*\brief Address of key in data storage. */
    size_t key_begin;
    /*\brief Address of end of key in data storage. */
    size_t key_end;
  };

  using iterator = ElemIterator<ValueImplT, false>;
  using const_iterator = ElemIterator<ValueImplT, true>;

 protected:
  void InitializeType(ValueKind kind) {
    CHECK(
        static_cast<uint8_t>(this->kind_) == static_cast<uint8_t>(ValueKind::kNull) ||
        static_cast<uint8_t>(this->kind_) == static_cast<uint8_t>(kind));
    CHECK(!finalised_) << "You can not change an existing value.";
    this->kind_ = kind;
  }

  void CheckType(ValueKind type) const {
    CHECK_EQ(static_cast<std::uint8_t>(type), static_cast<std::uint8_t>(this->kind_))
        << "kind: " << KindStr(type) << " self.kind: " << KindStr(this->kind_);
  }
  void CheckType(ValueKind a, ValueKind b) const {
    CHECK(static_cast<std::uint8_t>(a) == static_cast<std::uint8_t>(this->kind_) ||
               static_cast<std::uint8_t>(b) == static_cast<std::uint8_t>(this->kind_));
  }

  bool NeedFinalise() const {
    return !is_view_ && !finalised_ && (this->IsArray() || this->IsObject());
  }

 protected :

  struct StringStorage {
    size_t beg;
    size_t end;
  };

  // A simple wrapper for `std::vector<size_t>`, used when we are only viewing the object
  // (Dump).
  class ArrayTable {
    bool is_view_ {false};
    size_t size_;
    common::Span<size_t> tree_;
    std::vector<size_t> table_;
    size_t offset_;

   public:
    ArrayTable() = default;
    ArrayTable(common::Span<size_t> tree, size_t offset)
        : is_view_{true}, tree_{tree}, offset_{offset + 1} {
      size_ = tree_[offset];
    }
    size_t& operator[](size_t i) {
      if (is_view_) {
        return tree_[offset_ + i];
      }
      return table_[i];
    }
    size_t const& operator[](size_t i) const {
      if (is_view_) {
        return tree_[offset_ + i];
      }
      return table_[i];
    }
    size_t size() const {
      return size_;
    }

    void resize(size_t n) {
      CHECK(!is_view_);
      table_.resize(n);
      size_ = table_.size();
    }
    void resize(size_t n, size_t v) {
      CHECK(!is_view_);
      table_.resize(n, v);
      size_ = table_.size();
    }
    void reserve(size_t n) {
      CHECK(!is_view_);
      table_.reserve(n);
    }
    size_t length() const { return size_; }
    size_t *data() { return table_.data(); }
    void push_back(size_t v) {
      CHECK(!is_view_);
      table_.push_back(v);
      size_++;
    }
  };

  // explict working memory for non-trivial types.
  std::vector<ObjectElement> object_table_;
  ArrayTable array_table_;
  StringStorage  string_storage;

 protected:
  explicit ValueImpl(Container *doc) : handler_{doc}, self_{0} {
    handler_->Incref();
  }

  ValueImpl(Container *doc, size_t tree_beg)
      : handler_{doc}, self_{tree_beg}, kind_{ValueKind::kNull} {
    handler_->Incref();
  }

  // ValueImpl knows how to construct itself from data kind and a pointer to its storage
  ValueImpl(Container *doc, ValueKind kind, size_t self)
      : handler_{doc}, self_{self}, kind_{kind}, is_view_{true} {
    handler_->Incref();
    switch (kind_) {
    case ValueKind::kInteger: {
      this->kind_ = ValueKind::kInteger;
      break;
    }
    case ValueKind::kNumber: {
      this->kind_ = ValueKind::kNumber;
      break;
    }
    case ValueKind::kObject: {
      this->kind_ = ValueKind::kObject;
      common::Span<size_t> tree = handler_->Tree().Access();
      size_t table_offset = JsonTypeHandler::GetOffset(tree[self_]);
      size_t length = tree[table_offset];
      object_table_.resize(length);
      auto tree_ptr = table_offset + 1;
      for (size_t i = 0; i < length; ++i) {
        object_table_[i].key_begin = tree[tree_ptr];
        object_table_[i].key_end = tree[tree_ptr + 1];
        object_table_[i].value = tree[tree_ptr + 2];
        tree_ptr += 3;
      }
      break;
    }
    case ValueKind::kArray: {
      this->kind_ = ValueKind::kArray;
      common::Span<size_t> tree = handler_->Tree().Access();
      size_t table_offset = JsonTypeHandler::GetOffset(tree[self_]);
      array_table_ = ArrayTable(tree, table_offset);
      break;
    }
    case ValueKind::kNull: {
      this->kind_ = ValueKind::kNull;
      break;
    }
    case ValueKind::kString: {
      this->kind_ = ValueKind::kString;
      common::Span<size_t> tree = handler_->Tree().Access();
      auto storage_ptr = JsonTypeHandler::GetOffset(tree[self_]);
      string_storage.beg = tree[storage_ptr];
      string_storage.end = tree[storage_ptr + 1];
      break;
    }
    case ValueKind::kTrue: {
      this->kind_ = ValueKind::kTrue;
      break;
    }
    case ValueKind::kFalse: {
      this->kind_ = ValueKind::kFalse;
      break;
    }
    default: {
      LOG(FATAL) << "Invalid value type: " << static_cast<uint8_t>(kind_);
      break;
    }
    }
  }

  template <typename T>
  void SetNumber(T value, ValueKind kind) {
    InitializeType(kind);
    common::Span<size_t> tree = handler_->Tree().Access();
    auto const current_data_pointer = handler_->Data().Top();
    tree[this->self_] =
        JsonTypeHandler::MakeTypedOffset(current_data_pointer, kind);
    handler_->Data().Push(&value, 1);
  }

  template <typename T>
  T GetNumber(ValueKind kind) const {
    CheckType(kind);
    common::Span<size_t> tree = handler_->Tree().Access();
    auto data_ptr = JsonTypeHandler::GetOffset(tree[self_]);
    auto data = handler_->Data().Access();
    T value { T() };
    std::memcpy(&value, data.data() + data_ptr, sizeof(value));
    return value;
  }

  /*\brief Accept a writer for dump. */
  template <typename Writer> void Accept(Writer* writer) {
    // Here this object is assumed to be already initialized.
    switch (this->GetType()) {
    case ValueKind::kFalse: {
      writer->HandleFalse();
      break;
    }
    case ValueKind::kTrue: {
      writer->HandleTrue();
      break;
    }
    case ValueKind::kNull: {
      writer->HandleNull();
      break;
    }
    case ValueKind::kInteger: {
      writer->HandleInteger(this->GetInt());
      break;
    }
    case ValueKind::kNumber: {
      writer->HandleFloat(this->GetFloat());
      break;
    }
    case ValueKind::kString: {
      auto str = this->GetString();
      writer->HandleString(str);
      break;
    }
    case ValueKind::kArray: {
      writer->BeginArray();
      for (size_t it = 0; it < this->array_table_.size(); ++it) {
        if (array_table_[it] != kElementEnd) {
          auto value = this->GetArrayElem(it);
          value.Accept(writer);
        }
        if (it != this->array_table_.size() - 1) {
          writer->Comma();
        }
      }
      writer->EndArray();
      break;
    }
    case ValueKind::kObject: {
      writer->BeginObject();
      auto tree = handler_->Tree().Access();
      auto data = handler_->Data().Access();
      for (size_t i = 0; i < this->object_table_.size(); ++i) {
        ObjectElement const &elem = object_table_[i];
        ConstStringRef key(&(data[elem.key_begin]),
                           elem.key_end - elem.key_begin);
        writer->HandleString(key);

        writer->KeyValue();

        ValueKind kind = JsonTypeHandler::GetType(tree[elem.value]);
        ValueImpl value{handler_, kind, elem.value};
        value.Accept(writer);
        if (i != this->object_table_.size() - 1) {
          writer->Comma();
        }
      }
      writer->EndObject();
      break;
    }
    }
  }

 public:
  // Forbits copying as that incurs a copy of array/object table.
  ValueImpl(ValueImpl const &that) = delete;
  ValueImpl(ValueImpl &&that)
      : handler_{that.handler_}, self_{that.self_}, kind_{that.kind_},
        is_view_{that.is_view_}, finalised_{that.finalised_},
        object_table_{std::move(that.object_table_)},
        array_table_{std::move(that.array_table_)} {
    that.finalised_ = true;
    handler_->Incref();
  }

  virtual ~ValueImpl() {
    handler_->Decref();
    if (!this->NeedFinalise()) {
      return;
    }
    switch (this->GetType()) {
    case ValueKind::kObject: {
      this->EndObject();
      break;
    }
    case ValueKind::kArray: {
      this->EndArray();
      break;
    }
    default: {
      break;
    }
    }
  }

  bool IsObject() const { return kind_ == ValueKind::kObject; }
  bool IsArray() const { return kind_ == ValueKind::kArray; }
  bool IsString() const { return kind_ == ValueKind::kString; }

  bool IsInteger() const { return kind_ == ValueKind::kInteger; }
  bool IsNumber() const { return kind_ == ValueKind::kNumber; }
  bool IsTrue() const { return kind_ == ValueKind::kTrue; }
  bool IsFalse() const { return kind_ == ValueKind::kFalse; }
  bool IsNull() const { return kind_ == ValueKind::kNull; }

  size_t Length() const {
    CheckType(ValueKind::kArray, ValueKind::kObject);
    if (this->kind_ == ValueKind::kArray) {
      return this->array_table_.size();
    } else {
      return this->object_table_.size();
    }
  }

  ValueImpl& SetTrue() {
    InitializeType(ValueKind::kTrue);
    return *this;
  }

  ValueImpl& SetFalse() {
    InitializeType(ValueKind::kFalse);
    return *this;
  }

  ValueImpl& SetNull() {
    InitializeType(ValueKind::kNull);
    return *this;
  }

  ValueImpl& SetString(ConstStringRef string) {
    InitializeType(ValueKind::kString);
    StorageView<size_t> tree_storage = handler_->Tree();
    auto current_tree_pointer = tree_storage.Top();
    auto tree = tree_storage.Access();
    tree[this->self_] = JsonTypeHandler::MakeTypedOffset(
        current_tree_pointer, ValueKind::kString);

    tree_storage.Expand(2);
    tree = tree_storage.Access();

    StorageView<char> data_storage = handler_->Data();
    auto current_data_pointer = data_storage.Top();

    data_storage.Expand(string.size());
    auto data = data_storage.Access();
    std::memcpy(data.data() + current_data_pointer, string.data(), string.size());

    string_storage.beg = current_data_pointer;
    string_storage.end = current_data_pointer + string.size();

    tree[current_tree_pointer] = string_storage.beg;
    tree[current_tree_pointer + 1] = string_storage.end;
    return *this;
  }

  ValueImpl& operator=(ConstStringRef str) {
    return this->SetString(str);
  }
  void operator=(int64_t i) {
    this->SetInteger(i);
  }
  ValueImpl& operator=(float f) {
    this->SetFloat(f);
  }

  ConstStringRef GetString() const {
    CheckType(ValueKind::kString);
    auto data = handler_->Data().Access();
    return {&data[string_storage.beg], string_storage.end - string_storage.beg};
  }

  void SetInteger(int64_t i) {
    SetNumber(i, ValueKind::kInteger);
  }

  void SetFloat(float f) {
    SetNumber(f, ValueKind::kNumber);
  }
  /*\brief Get floating point value. */
  float GetFloat() const {
    return GetNumber<float>(ValueKind::kNumber);
  }
  /*\brief Get integer value. */
  int64_t GetInt() const {
    return GetNumber<int64_t>(ValueKind::kInteger);
  }

  /*\brief Set this value to an array with pre-defined length. */
  ValueImpl& SetArray(size_t length) {
    InitializeType(ValueKind::kArray);
    common::Span<size_t> tree = handler_->Tree().Access();
    tree[self_] = JsonTypeHandler::MakeTypedOffset(kElementEnd,
                                                         ValueKind::kArray);
    array_table_.resize(length, kElementEnd);
    finalised_ = false;
    return *this;
  }
  /*\brief Hint the array size. */
  void SizeHint(size_t n) {
    CheckType(ValueKind::kArray);
    array_table_.reserve(n);
  }
  /*\brief Get one array element by index. */
  ValueImpl GetArrayElem(size_t index) {
    CheckType(ValueKind::kArray);
    CHECK_LT(index, array_table_.size());
    StorageView<size_t> tree_storage = handler_->Tree();
    auto tree = tree_storage.Access();
    ValueKind kind;
    if (array_table_[index] == kElementEnd) {
      auto value = ValueImpl {handler_, tree_storage.Top()};
      array_table_[index] = tree_storage.Top();
      tree_storage.Expand(1);
      return value;
    } else {
      kind = JsonTypeHandler::GetType(tree[array_table_[index]]);
      ValueImpl value(handler_, kind, array_table_[index]);
      return value;
    }
  }
  ValueImpl GetArrayElem(size_t index) const {
    CheckType(ValueKind::kArray);
    CHECK_LT(index, array_table_.size());
    StorageView<size_t> tree_storage = handler_->Tree();
    auto tree = tree_storage.Access();
    ValueKind kind = JsonTypeHandler::GetType(tree[array_table_[index]]);
    ValueImpl value(handler_, kind, array_table_[index]);
    return value;
  }
  /*\brief Set this value to be an array without providing the length of array. */
  ValueImpl& SetArray() {
    this->SetArray(0);
    return *this;
  }
  /*\brief Add a new element to array. */
  ValueImpl CreateArrayElem() {
    CheckType(ValueKind::kArray);
    StorageView<size_t> tree_storage = handler_->Tree();
    auto value = ValueImpl {handler_, tree_storage.Top()};
    array_table_.push_back(tree_storage.Top());
    tree_storage.Expand(1);
    return value;
  }
  /*\brief Set this value to be an object. */
  ValueImpl& SetObject() {
    // initialize the value here.
    InitializeType(ValueKind::kObject);
    auto tree = handler_->Tree().Access();
    tree[self_] =
        JsonTypeHandler::MakeTypedOffset(self_, ValueKind::kObject);
    finalised_ = false;
    return *this;
  }
  /*\brief Create a member in object. */
  ValueImpl CreateMember(ConstStringRef key) {
    CheckType(ValueKind::kObject);
    StorageView<size_t> tree_storage = handler_->Tree();
    // allocate space for object element.
    tree_storage.Expand(sizeof(ObjectElement) / sizeof(size_t) + 1);
    auto tree = tree_storage.Access();

    auto data_storage = handler_->Data();
    size_t const current_data_pointer =  data_storage.Top();
    // allocate space for key
    data_storage.Expand(key.size() * sizeof(ConstStringRef::value_type));
    auto data = data_storage.Access();
    std::copy(key.cbegin(), key.cend(), data.begin() + current_data_pointer);

    ObjectElement elem;
    elem.key_begin = current_data_pointer;
    elem.key_end = current_data_pointer + key.size();
    elem.value = tree_storage.Top() - 1;

    object_table_.push_back(elem);

    auto value = ValueImpl{handler_, elem.value};
    return value;
  }
  /*\brief Finish creating object.  This is called upon destruction. */
  void EndObject() {
    CheckType(ValueKind::kObject);
    CHECK(!finalised_);
    StorageView<size_t> tree_storage = handler_->Tree();
    size_t current_tree_pointer = tree_storage.Top();
    auto table_begin = current_tree_pointer;

    tree_storage.Expand(object_table_.size() * 3 + 1);  // +1 length
    auto tree = tree_storage.Access();
    tree[current_tree_pointer] = object_table_.size();
    current_tree_pointer += 1;

    for (size_t i = 0; i < object_table_.size(); ++i) {
      tree[current_tree_pointer] = object_table_[i].key_begin;
      tree[current_tree_pointer + 1] = object_table_[i].key_end;
      tree[current_tree_pointer + 2] = object_table_[i].value;
      current_tree_pointer += 3;
    }

    tree[self_] = JsonTypeHandler::MakeTypedOffset(
        table_begin, ValueKind::kObject);
    finalised_ = true;
  }
  /*\brief Finish creating array.  This is called upon destruction. */
  void EndArray() {
    CheckType(ValueKind::kArray);
    CHECK(!finalised_);
    StorageView<size_t> tree_storage = handler_->Tree();
    size_t current_tree_pointer = tree_storage.Top();
    auto table_begin = current_tree_pointer;

    tree_storage.Expand(array_table_.size() + 1);
    auto tree = tree_storage.Access();
    tree[current_tree_pointer] = array_table_.size();
    current_tree_pointer += 1;

    std::memcpy(tree.data() + current_tree_pointer,
                array_table_.data(), array_table_.size() * sizeof(size_t));
    tree[self_] = JsonTypeHandler::MakeTypedOffset(
        table_begin, ValueKind::kArray);
    finalised_ = true;
  }
  ConstStringRef GetKeyByIndex(size_t index) const {
    CheckType(ValueKind::kObject);
    CHECK_LT(index, object_table_.size());
    StorageView<size_t> tree_storage = handler_->Tree();
    auto tree = tree_storage.Access();
    auto elem = object_table_[index];
    auto data = handler_->Data().Access();
    return ConstStringRef(&data[elem.key_begin], elem.key_end - elem.key_begin);
  }
  /*\brief Get an object member by its index, similar to std::map::at. Used by iterator.*/
  ValueImplT GetMemberByIndex(size_t index) const {
    CheckType(ValueKind::kObject);
    CHECK_LT(index, object_table_.size());
    StorageView<size_t> tree_storage = handler_->Tree();
    auto tree = tree_storage.Access();
    auto elem = object_table_[index];
    ValueKind kind = JsonTypeHandler::GetType(tree[elem.value]);
    ValueImpl value{handler_, kind, elem.value};
    return value;
  }
  /*\brief Find a object member by its key. */
  const_iterator FindMemberByKey(ConstStringRef key) const {
    CheckType(ValueKind::kObject);
    common::Span<size_t> tree = handler_->Tree().Access();
    auto data = handler_->Data().Access();
    for (size_t i = 0; i < object_table_.size(); ++i) {
      auto elem = object_table_[i];
      char const* c_str = &data[elem.key_begin];
      auto size = elem.key_end - elem.key_begin;
      auto str = ConstStringRef(c_str, size);
      bool equal = false;
      if (str.size() == key.size()) {
        equal = std::memcmp(c_str, key.data(), str.size()) == 0;
      } else {
        equal = false;
      }

      if (equal) {
        return const_iterator(this, i);
      }
    }
    return const_iterator {this, object_table_.size()};
  }

  const_iterator cbegin() const {  // NOLINT
    CheckType(ValueKind::kObject);
    return const_iterator {this, 0};
  }
  const_iterator cend() const {  // NOLINT
    CheckType(ValueKind::kObject);
    return const_iterator {this, object_table_.size()};
  }
  iterator begin() {
    CheckType(ValueKind::kObject);
    return iterator {this, 0};
  }
  iterator end() {
    CheckType(ValueKind::kObject);
    return iterator {this, object_table_.size()};
  }
  const_iterator begin() const {
    return this->cbegin();
  }
  const_iterator end() const {
    return this->cend();
  }

  ValueKind GetType() const {
    return this->kind_;
  }
};

enum class jError : std::uint8_t {
  kSuccess = 0,
  kInvalidNumber,
  kInvalidArray,
  kInvalidObject,
  kUnexpectedEnd,
  kInvalidString,
  kInvalidTrue,
  kInvalidFalse,
  kInvalidNull,
  kEmptyInput,
  kUnknownConstruct
};

/*!
 * \brief JSON document root.  Also is a container for all other derived objects, so it
 * must be the last one to go out of scope when using the JSON tree.  For now only using
 * object as root is supported.
 *
 * Example usage:
 *
 *   Document doc = Document::Load(str);
 *   Json value = doc.FindMemberByKey("Learner");
 *
 *   auto str = doc.Dump();
 */
class Document {
  jError err_code_ { jError::kSuccess };

  std::vector<size_t> _tree_storage;
  std::vector<std::string::value_type> _data_storage;

  int32_t n_alive_values_;

  using Value = ValueImpl<Document>;
  Value value;
  /*\brief Last character used in reporting parsing error. */
  size_t last_character;

  StorageView<size_t> Tree() {
    return StorageView<size_t>{&_tree_storage};
  }
  StorageView<char> Data() {
    return StorageView<char> {&_data_storage};
  }
  friend Value;

  class GlobalCLocale {
    std::locale ori_;

   public:
    GlobalCLocale() : ori_{std::locale()} {
      std::string const name{"C"};
      try {
        std::locale::global(std::locale(name.c_str()));
      } catch (std::runtime_error const &e) {
        LOG(FATAL) << "Failed to set locale: " << name;
      }
    }
    ~GlobalCLocale() { std::locale::global(ori_); }
  };

  void AssertValidExit() {
    CHECK_EQ(n_alive_values_, 1) << "All values must go out of scope before Document.";
    CHECK(value.finalised_);
  }

 private:
  explicit Document(bool empty) :
      n_alive_values_ {0},
      value(this),
      last_character{0} {
    this->_tree_storage.resize(1);
  }

 public:
  Document() : n_alive_values_ {0}, value(this), last_character{0} {
    // right now document root must be an object.
    this->_tree_storage.resize(1);
    this->value.SetObject();
  }
  explicit Document(ValueKind kind) :
      n_alive_values_ {0},
      value(this),
      last_character {0} {
    this->_tree_storage.resize(1);
    switch (kind) {
      case ValueKind::kArray: {
        this->value.SetArray();
        break;
      }
      case ValueKind::kObject: {
        this->value.SetObject();
        break;
      }
      default: {
        LOG(FATAL) << "Invalid value type for document root.";
      }
    }
  }
  Document(Document const& that) = delete;
  Document(Document&& that) :
      err_code_{that.err_code_},
      _tree_storage{std::move(that._tree_storage)},
      _data_storage{std::move(that._data_storage)},
      n_alive_values_ {0},
      value{ValueImpl<Document>{this}},
      last_character{that.last_character} {
        that.value.finalised_ = {true};
        value.object_table_ = std::move(that.value.object_table_);
        value.array_table_ = std::move(that.value.array_table_);
        value.kind_ = that.value.kind_;
        CHECK(value.IsObject());
      }

  ~Document() {
    if (!value.finalised_ && err_code_ == jError::kSuccess) {
      if (value.IsObject()) {
        value.EndObject();
      } else {
        value.EndArray();
      }
    }
    AssertValidExit();
  }

  Value CreateMember(ConstStringRef key) {
    return this->value.CreateMember(key);
  }

  Value& GetObject() {
    return value;
  }
  Value const &GetObject() const {
    return value;
  }
  Value& GetValue() {
    return value;
  }
  Value const& GetValue() const {
    return value;
  }

  void Decref() { this->n_alive_values_--; }
  void Incref() { this->n_alive_values_++; }

  size_t Length() const {
    return this->value.Length();
  }

  jError Errc() const {
    return err_code_;
  }
  std::string ErrStr() const {
    std::string msg;
    switch (err_code_) {
      case jError::kSuccess:
        return "Success";
      case jError::kInvalidNumber:
        msg = "Found invalid floating point number.";
        break;
      case jError::kInvalidArray:
        msg =  "Found invalid array structure.";
      case jError::kInvalidObject:
        msg = "Found invalid object structure.";
        break;
      case jError::kInvalidString:
        msg = "Found invalid string.";
        break;
      case jError::kUnexpectedEnd:
        msg = "Unexpected end.";
        break;
      case jError::kInvalidTrue:
        msg = "Error occurred while parsing `true'.";
        break;
      case jError::kInvalidFalse:
        msg = "Error occurred while parsing `false'.";
        break;
      case jError::kInvalidNull:
        msg = "Found invalid `null'.";
        break;
      case jError::kEmptyInput:
        msg = "Empty input string is not allowed.";
        break;
      case jError::kUnknownConstruct:
        msg = "Unknown construct.";
        break;
      default:
        LOG(FATAL) << "Unknown error code";
    }
    msg += " at character:" + std::to_string(last_character);
    return msg;
  }

  template <typename Reader>
  static Document Load(StringRef json_str) {
    // only used in loading as slow path for reading float is `std::strtod', while any
    // other place has dedicated implementation.
    GlobalCLocale guard;
    Document doc(false);
    doc._tree_storage.reserve(json_str.size() * 2);
    doc._data_storage.reserve(json_str.size() * 2);

    if (json_str.size() == 0) {
      doc.err_code_ = jError::kEmptyInput;
      doc.value.finalised_ = true;
      return doc;
    }

    Reader reader(json_str, &(doc.value));
    std::tie(doc.err_code_, doc.last_character) = reader.Parse();
    CHECK(doc.value.GetType() == ValueKind::kObject ||
          doc.value.GetType() == ValueKind::kArray);

    return doc;
  }
  template <typename Writer>
  std::string Dump() {
    CHECK(err_code_ == jError::kSuccess);
    if (!value.finalised_) {
      switch (value.kind_) {
        case ValueKind::kObject: {
          value.EndObject();
          break;
        }
        case ValueKind::kArray: {
          value.EndArray();
          break;
        }
        default: {
          LOG(FATAL) << "Invalid value type for document root.";
        }
      }
    }
    AssertValidExit();
    std::string result;
    Writer writer;

    this->value.Accept(&writer);
    writer.TakeResult(&result);

    return result;
  }
};

using Json = ValueImpl<Document>;

template <typename P>
void toJson(Json* p_out, P const& parameter) {
  p_out->SetObject();
  for (auto const& kv : parameter.__DICT__()) {
    auto key = p_out->CreateMember(kv.first);
    key.SetString(kv.second);
  }
}

template <typename P>
void fromJson(P *parameter,  Json const& in) {
  std::map<std::string, std::string> m;
  for (auto it = in.cbegin(); it != in.cend(); ++it) {
    m[it.Key().data()] = (*it).GetString().data();
  }
  parameter->UpdateAllowUnknown(m);
}
}  // namespace experimental
}  // namespace xgboost
#endif  // XGBOOST_COMMON_JSON_EXPERIMENTAL_H_
