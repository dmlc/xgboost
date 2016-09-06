/*!
 *  Copyright (c) 2015 by Contributors
 * \file registry.h
 * \brief Registry utility that helps to build registry singletons.
 */
#ifndef DMLC_REGISTRY_H_
#define DMLC_REGISTRY_H_

#include <map>
#include <string>
#include <vector>
#include "./base.h"
#include "./logging.h"
#include "./parameter.h"
#include "./type_traits.h"

namespace dmlc {
/*!
 * \brief Registry class.
 *  Registry can be used to register global singletons.
 *  The most commonly use case are factory functions.
 *
 * \tparam EntryType Type of Registry entries,
 *     EntryType need to name a name field.
 */
template<typename EntryType>
class Registry {
 public:
  /*! \return list of functions in the registry */
  inline static const std::vector<const EntryType*> &List() {
    return Get()->entry_list_;
  }
  /*!
   * \brief Find the entry with corresponding name.
   * \param name name of the function
   * \return the corresponding function, can be NULL
   */
  inline static const EntryType *Find(const std::string &name) {
    const std::map<std::string, EntryType*> &fmap = Get()->fmap_;
    typename std::map<std::string, EntryType*>::const_iterator p = fmap.find(name);
    if (p != fmap.end()) {
      return p->second;
    } else {
      return NULL;
    }
  }
  /*!
   * \brief Internal function to register a name function under name.
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  inline EntryType &__REGISTER__(const std::string& name) {
    CHECK_EQ(fmap_.count(name), 0)
        << name << " already registered";
    EntryType *e = new EntryType();
    e->name = name;
    fmap_[name] = e;
    entry_list_.push_back(e);
    return *e;
  }
  /*!
   * \brief Internal function to either register or get registered entry
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  inline EntryType &__REGISTER_OR_GET__(const std::string& name) {
    if (fmap_.count(name) == 0) {
      return __REGISTER__(name);
    } else {
      return *fmap_.at(name);
    }
  }
  /*!
   * \brief get a singleton of the Registry.
   *  This function can be defined by DMLC_ENABLE_REGISTRY.
   * \return get a singleton
   */
  static Registry *Get();

 private:
  /*! \brief list of entry types */
  std::vector<const EntryType*> entry_list_;
  /*! \brief map of name->function */
  std::map<std::string, EntryType*> fmap_;
  /*! \brief constructor */
  Registry() {}
  /*! \brief destructor */
  ~Registry() {
    for (typename std::map<std::string, EntryType*>::iterator p = fmap_.begin();
         p != fmap_.end(); ++p) {
      delete p->second;
    }
  }
};

/*!
 * \brief Common base class for function registry.
 *
 * \code
 *  // This example demonstrates how to use Registry to create a factory of trees.
 *  struct TreeFactory :
 *      public FunctionRegEntryBase<TreeFactory, std::function<Tree*()> > {
 *  };
 *
 *  // in a independent cc file
 *  namespace dmlc {
 *  DMLC_REGISTRY_ENABLE(TreeFactory);
 *  }
 *  // register binary tree constructor into the registry.
 *  DMLC_REGISTRY_REGISTER(TreeFactory, TreeFactory, BinaryTree)
 *      .describe("Constructor of BinaryTree")
 *      .set_body([]() { return new BinaryTree(); });
 * \endcode
 *
 * \tparam EntryType The type of subclass that inheritate the base.
 * \tparam FunctionType The function type this registry is registerd.
 */
template<typename EntryType, typename FunctionType>
class FunctionRegEntryBase {
 public:
  /*! \brief name of the entry */
  std::string name;
  /*! \brief description of the entry */
  std::string description;
  /*! \brief additional arguments to the factory function */
  std::vector<ParamFieldInfo> arguments;
  /*! \brief Function body to create ProductType */
  FunctionType body;
  /*! \brief Return type of the function */
  std::string return_type;

  /*!
   * \brief Set the function body.
   * \param body Function body to set.
   * \return reference to self.
   */
  inline EntryType &set_body(FunctionType body) {
    this->body = body;
    return this->self();
  }
  /*!
   * \brief Describe the function.
   * \param description The description of the factory function.
   * \return reference to self.
   */
  inline EntryType &describe(const std::string &description) {
    this->description = description;
    return this->self();
  }
  /*!
   * \brief Add argument information to the function.
   * \param name Name of the argument.
   * \param type Type of the argument.
   * \param description Description of the argument.
   * \return reference to self.
   */
  inline EntryType &add_argument(const std::string &name,
                                 const std::string &type,
                                 const std::string &description) {
    ParamFieldInfo info;
    info.name = name;
    info.type = type;
    info.type_info_str = info.type;
    info.description = description;
    arguments.push_back(info);
    return this->self();
  }
  /*!
   * \brief Append list if arguments to the end.
   * \param args Additional list of arguments.
   * \return reference to self.
   */
  inline EntryType &add_arguments(const std::vector<ParamFieldInfo> &args) {
    arguments.insert(arguments.end(), args.begin(), args.end());
    return this->self();
  }
  /*!
  * \brief Set the return type.
  * \param type Return type of the function, could be Symbol or Symbol[]
  * \return reference to self.
  */
  inline EntryType &set_return_type(const std::string &type) {
    return_type = type;
    return this->self();
  }

 protected:
  /*!
   * \return reference of self as derived type
   */
  inline EntryType &self() {
    return *(static_cast<EntryType*>(this));
  }
};

/*!
 * \def DMLC_REGISTRY_ENABLE
 * \brief Macro to enable the registry of EntryType.
 * This macro must be used under namespace dmlc, and only used once in cc file.
 * \param EntryType Type of registry entry
 */
#define DMLC_REGISTRY_ENABLE(EntryType)                                 \
  template<>                                                            \
  Registry<EntryType > *Registry<EntryType >::Get() {                   \
    static Registry<EntryType > inst;                                   \
    return &inst;                                                       \
  }                                                                     \

/*!
 * \brief Generic macro to register an EntryType
 *  There is a complete example in FactoryRegistryEntryBase.
 *
 * \param EntryType The type of registry entry.
 * \param EntryTypeName The typename of EntryType, must do not contain namespace :: .
 * \param Name The name to be registered.
 * \sa FactoryRegistryEntryBase
 */
#define DMLC_REGISTRY_REGISTER(EntryType, EntryTypeName, Name)          \
  static DMLC_ATTRIBUTE_UNUSED EntryType & __make_ ## EntryTypeName ## _ ## Name ## __ =      \
      ::dmlc::Registry<EntryType>::Get()->__REGISTER__(#Name)           \

/*!
 * \brief (Optional) Declare a file tag to current file that contains object registrations.
 *
 *  This will declare a dummy function that will be called by register file to
 *  incur a link dependency.
 *
 * \param UniqueTag The unique tag used to represent.
 * \sa DMLC_REGISTRY_LINK_TAG
 */
#define DMLC_REGISTRY_FILE_TAG(UniqueTag)                                \
  int __dmlc_registry_file_tag_ ## UniqueTag ## __() { return 0; }

/*!
 * \brief (Optional) Force link to all the objects registered in file tag.
 *
 *  This macro must be used in the same file as DMLC_REGISTRY_ENABLE and
 *  in the same namespace as DMLC_REGISTRY_FILE_TAG
 *
 *  DMLC_REGISTRY_FILE_TAG and DMLC_REGISTRY_LINK_TAG are optional macros for registration.
 *  They are used to encforce link of certain file into during static linking.
 *
 *  This is mainly used to solve problem during statically link a library which contains backward registration.
 *  Specifically, this avoids the objects in these file tags to be ignored by compiler.
 *
 *  For dynamic linking, this problem won't occur as everything is loaded by default.
 *
 *  Use of this is optional as it will create an error when a file tag do not exist.
 *  An alternative solution is always ask user to enable --whole-archieve during static link.
 *
 * \begincode
 * // in file objective_registry.cc
 * DMLC_REGISTRY_ENABLE(MyObjective);
 * DMLC_REGISTRY_LINK_TAG(regression_op);
 * DMLC_REGISTRY_LINK_TAG(rank_op);
 *
 * // in file regression_op.cc
 * // declare tag of this file.
 * DMLC_REGISTRY_FILE_TAG(regression_op);
 * DMLC_REGISTRY_REGISTER(MyObjective, logistic_reg, logistic_reg);
 * // ...
 *
 * // in file rank_op.cc
 * // declare tag of this file.
 * DMLC_REGISTRY_FILE_TAG(rank_op);
 * DMLC_REGISTRY_REGISTER(MyObjective, pairwiserank, pairwiserank);
 *
 * \endcode
 *
 * \param UniqueTag The unique tag used to represent.
 * \sa DMLC_REGISTRY_ENABLE, DMLC_REGISTRY_FILE_TAG
 */
#define DMLC_REGISTRY_LINK_TAG(UniqueTag)                                \
  int __dmlc_registry_file_tag_ ## UniqueTag ## __();                   \
  static int DMLC_ATTRIBUTE_UNUSED __reg_file_tag_ ## UniqueTag ## __ = \
      __dmlc_registry_file_tag_ ## UniqueTag ## __();
}  // namespace dmlc
#endif  // DMLC_REGISTRY_H_
