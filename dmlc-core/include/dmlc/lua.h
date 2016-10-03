/*!
 *  Copyright (c) 2016 by Contributors
 * \file lua.h
 * \brief C++11 header only interface to easily interact with Lua and Torch.
 *  This code is evolved from torch plugin code for MXNet.
 *
 *  This header will require Torch and Lua to be presented, do not include.
 *
 * \author Junyuan Xie, Min Lin, Tianqi Chen
 *
 * \code
 *
 * // Example code to use the lua module.
 * dmlc::LuaState* lua = dmlc::LuaState::ThreadLocalState();
 * // vectors converts automatically to lua table.
 * auto tbl = lua->Convert(std::vector<int>{1,2,3});
 * // use eval to get lua reference, this is a function
 * auto print = lua->Eval("return function(x) print(x) end");
 * // lua function can be directly called from c++, arguments are converted.
 * print(100);
 *
 * // set field in the table.
 * tbl.SetField("square", lua->Eval("return function(x) x*x end"));
 * // call the function, covert back to C++ values.
 * int x = tbl["square"](100).Get<int>();
 *
 * \endcode
 */
#ifndef DMLC_LUA_H_
#define DMLC_LUA_H_

extern "C" {
#include <lua.h>
#include <luaT.h>
#include <lualib.h>
}

#include <string>
#include <stdexcept>
#include <tuple>
#include <mutex>
#include <memory>
#include <vector>
#include <utility>
#include <unordered_map>
#include <type_traits>

#include "./base.h"
#include "./logging.h"
#include "./thread_local.h"

namespace dmlc {

// forward declare torch state
class LuaState;

namespace lua_stack {
template<typename T>
struct Handler;
};

/*! \brief an reference to lua object */
class LuaRef {
 public:
  /*! \brief construct an nil ref */
  LuaRef() = default;
  /*!
   * \brief move constructor from another LuaRef
   * \param other The other LuaRef to be moved
   */
  inline LuaRef(LuaRef&& other);  // NOLINT(*)
  /*!
   * \brief copy constructor
   * \param other The other LuaRef to be copied
   */
  inline LuaRef(const LuaRef& other);  // NOLINT(*)
  /*!
   * \brief assign operator from other
   * \param other The other LuaRef to be copy or moved.
   * \return self
   */
  inline LuaRef& operator=(LuaRef&& other);
  /*!
   * \brief assign operator from other
   * \param other The other LuaRef to be copy or moved.
   * \return self
   */
  inline LuaRef& operator=(const LuaRef& other);
  /*! \brief destructor */
  inline ~LuaRef();
  /*!
   * \brief swap content with another ref
   * \param other another LuaRef to be swaped.
   */
  inline void swap(LuaRef& other); // NOLINT(*)
  /*!
   * \brief Get content out as type T.
   *
   * \tparam T the type to be fetched.
   * \return the corresponding c type.
   */
  template<typename T>
  inline T Get() const;
  /*!
   * \brief Get user data pointer from LuaRef
   *
   *  CAREFUL when getting userdata(e.g. pointer to Tensor's storage) from LuaRef.
   *  Remember they are managed by Lua, and can get deleted when all the
   *  LuaRef to the userdata destructs. A good practice is always use a LuaRef to keep
   *  the userdata alive when you need them from C++ side.
   *
   * \tparam T the type of pointer to be fetched.
   * \return the corresponding c type.
   */
  template<typename T>
  inline T* GetUDataPtr() const;
  /*! \return whether the value is nil */
  inline bool is_nil() const;
  /*!
   * \brief invoke the LuaRef as function
   * \param args Arguments to be passed.
   * \tparam Args arguments to be passed.
   * \return The first return value.
   */
  template<typename... Args>
  inline LuaRef operator()(Args&& ...args) const;
  /*!
   * \brief Get field from the lua table.
   *  The reference must be a table
   * \param key The key to the table
   * \return a new ref to the corresponding field.
   */
  inline LuaRef operator[](const std::string& key) const;
  /*!
   * \brief Get field from the lua array
   *  The reference must be a array
   * \param index The index to the array,
   *  Note: the index convention follows lua table, starts from 1
   * \return a new ref to the corresponding field.
   */
  inline LuaRef operator[](size_t index) const;
  /*!
   * \brief Set field of lua table.
   *  The reference must be a table
   * \param key The key to the table
   * \param value Lua convertable value to be setted.
   * \return self.
   */
  template<typename T>
  inline LuaRef& SetField(const std::string& key, const T& value);  // NOLINT(*)
  /*!
   * \brief Set LuaRef to the value on top of the stack.
   *  This state must be nil.
   *  This is API used by developer.
   *
   * \param s the corresponding lua state.
   */
  inline void SetByPopStack_(LuaState* s);

 private:
  // friend with luastate
  friend struct lua_stack::Handler<LuaRef>;
  friend class LuaState;
  friend std::ostream &operator<<(std::ostream &os, const LuaRef &r);
  /*! \brief pointer to the state */
  LuaState* state_{nullptr};
  /*! \brief reference index */
  int ref_;
};

/*! \brief A Lua state */
class LuaState {
 public:
  /*! \brief options to be provided in lua state */
  enum Option {
    kNoThreadProtect,
    kThreadLocal,
    kLocking,
  };
  /*! \brief destructor */
  inline ~LuaState();
  /*!
   * \brief evaluate a piece of lua code, return the first result.
   * \param lua_code Lua code
   * \return A LuaRef object of the first returned result,
   *  Can be nil if the code did not return LuaRefthing.
   */
  inline LuaRef Eval(const char* lua_code);
  /*!
   * \brief evaluate a piece of lua code, return the first result.
   * \param lua_code Lua code
   * \return A LuaRef object of the first returned result,
   *  Can be nil if the code did not return anything.
   */
  inline LuaRef Eval(const std::string& lua_code) {
    return this->Eval(lua_code.c_str());
  }
  /*!
   * \brief convert a C++ type to lua type
   * \param value The data to be converted.
   *  vector, map will be converted to table.
   * \return a converted value.
   * \tparam T the type to be converted.
   */
  template<typename T>
  inline LuaRef Convert(const T& value);
  /*!
   * \brief get global field from the state
   * \param key The key to the global field.
   * \return The global field value.
   */
  inline LuaRef operator[](const std::string& key);
  /*!
   * \brief Set the value to the global table.
   * \param key The key of the global field.
   * \param value The value to the set.
   */
  inline void SetGlobalField(const std::string& key, const LuaRef& value);
  /*!
   *  Get a thread local version of lua state.
   *  The LuaState runs in thread local mode,
   *  all the LuaRef can only be run on the current thread.
   *  This is the recommended behavior when invoking Lua.
   *
   * \return a threadlocal version of lua state.
   */
  static inline LuaState* ThreadLocalState();
  /*!
   * Create a new lua state.
   * \note It is highly recommended to use ThreadLocalState instead.
   *
   *  Most Lua program assumes it only runs from the same thread.
   *  Some Lua code that wraps C library(e.g. Torch) could rely
   *  on thread_local storage to store global state such as random number generator.
   *  This means if the code is invoked by another thread, the thread_local
   *  might become inavailable, depending on the implementation.
   *
   *  If the global state is stored only in Lua's global table, then
   *  it is safe to use kLocking mode and call the code from multiple thread.
   *  Never-the-less, using ThreadLocalState removes the need to lock,
   *  and is the desirable usecase in most times.
   *
   * \sa ThreadLocalState
   * \param option The option to use the state.
   * \return a newly created lua state
   */
  static inline LuaState* Create_(Option option);

  /*!
   * \brief protected run f, this is used by API developers.
   *  always call this to access lua state
   *  f must not destruct LuaRef, or access the mutex
   *
   * \param f the function to be called.
   * \tparam F the function to be called, signiture (lua_State *L)
   */
  template<typename F>
  inline void PRun_(F f);
  /*!
   * \param L the other lua state.
   * \return if the internal lua state is same as L
   */
  inline bool SameLuaState(lua_State *L) const {
    return L_ == L;
  }

 protected:
  struct StackReset;
  friend class LuaRef;
  friend struct ThreadLocalStore<LuaState>;
  /*!
   * \brief constructor
   */
  inline LuaState();

  /*! \brief internal option, default to thread local */
  Option option_{kThreadLocal};
  /*! \brief internal lua state */
  lua_State* L_;
  /*! \brief internal lock about the state */
  std::mutex mutex_;
};

// implementations after this line
//! \cond Doxygen_Suppress
/*! \brief macro to check error during lua call */
#define LUA_CALL(x)                                                     \
  if ((x)) {                                                            \
    LOG(FATAL) << "Lua Call Error:" <<  lua_tostring(L, -1);            \
  }

/*!
 * \brief namespace to handle conversions between lua and c++
 *  User can provide an specialization of dmlc::lua_stack::Handler
 *  to allow customized c++ data types to interact with Lua.
 *
 *  By default basic data types, composition of vector, and unordered_map is supported.
 *  The conversion rules
 *  - basic types(string, int, float) to corresponding lua types.
 *  - unordered_map to Lua table.
 *  - vector to lua indexed table.
 */
namespace lua_stack {
inline int lua_abs_index(lua_State* L, int index) {
  if (index > 0 || index <= LUA_REGISTRYINDEX) return index;
  return lua_gettop(L) + index + 1;
}

template<typename T>
struct Handler;

template<typename T>
struct NumberHandler {
  static inline T Get(lua_State* L, int index, LuaState* s) {
    CHECK_EQ(lua_type(L, index), LUA_TNUMBER)
        << "Attempt to get number but type is \'"
        << lua_typename(L, lua_type(L, index)) << '\'';
    if (std::is_integral<T>::value) {
      return static_cast<T>(lua_tointeger(L, index));
    } else {
      return static_cast<T>(lua_tonumber(L, index));
    }
  }
  static inline void Push(lua_State* L, const T& v) {
    if (std::is_integral<T>::value) {
      lua_pushinteger(L, static_cast<lua_Integer>(v));
    } else {
      lua_pushnumber(L, static_cast<lua_Number>(v));
    }
  }
};

template<typename ContainerType>
struct MapHandler {
  using K = typename ContainerType::key_type;
  using V = typename ContainerType::mapped_type;
  static inline ContainerType Get(lua_State* L, int index, LuaState* s) {
    ContainerType ret;
    CHECK(lua_istable(L, index))
        << "Expected a table but get "
        << lua_typename(L, lua_type(L, index)) << '\'';
    int tid = lua_abs_index(L, index);
    lua_pushnil(L);
    while (lua_next(L, -2)) {
      ret[Handler<K>::Get(L, -2, s)] = Handler<V>::Pop(L, -1, s);
      lua_pop(L, 1);
    }
    lua_settop(L, tid);
    return ret;
  }
  static inline void Push(lua_State* L, const ContainerType& v) {
    lua_createtable(L, v.size(), 0);
    for (const auto& kv : v) {
      Handler<K>::Push(L, kv.first);
      Handler<V>::Push(L, kv.second);
      lua_settable(L, -3);
    }
  }
};

struct UndefinedHandler {
};

template<typename T>
struct Handler
    : public std::conditional<std::is_arithmetic<T>::value,
                              NumberHandler<T>,
                              UndefinedHandler>::type {
};

template<>
struct Handler<std::string> {
  static inline std::string Get(lua_State* L, int index, LuaState* s) {
    CHECK_EQ(lua_type(L, index), LUA_TSTRING);
    return std::string(lua_tostring(L, index));
  }
  static inline void Push(lua_State* L, const std::string& v) {
    lua_pushstring(L, v.c_str());
  }
};

template<typename T>
struct Handler<std::vector<T> > {
  static inline std::vector<T> Get(lua_State* L, int index, LuaState* s) {
    std::vector<T> ret;
    CHECK(lua_istable(L, index))
        << "Expected a table but get "
        << lua_typename(L, lua_type(L, index)) << '\'';
    int tid = lua_abs_index(L, index);
    lua_pushnil(L);
    while (lua_next(L, tid)) {
      CHECK_EQ(Handler<size_t>::Get(L, -2, s), ret.size() + 1)
          << "Target table is not an array";
      ret.push_back(Handler<T>::Get(L, -1, s));
      lua_pop(L, 1);
    }
    lua_settop(L, tid);
    return ret;
  }
  static inline void Push(lua_State* L, const std::vector<T>& v) {
    lua_createtable(L, v.size(), 0);
    for (size_t i = 0; i < v.size(); ++i) {
      Handler<T>::Push(L, v[i]);
      lua_rawseti(L, -2, i + 1);
    }
  }
};

template<typename K, typename V>
struct Handler<std::unordered_map<K, V> >
    : public MapHandler<std::unordered_map<K, V> > {
};

template<>
struct Handler<LuaRef> {
  static inline LuaRef Get(lua_State* L, int index, LuaState* s) {
    LuaRef ret;
    lua_pushvalue(L, index);
    ret.SetByPopStack_(s);
    return ret;
  }

  static inline void Push(lua_State* L, const LuaRef& v) {
    if (v.is_nil()) {
      lua_pushnil(L);
    } else {
      CHECK(v.state_->SameLuaState(L))
          << "Cannot pass LuaRef on a different LuaState's function";
      lua_rawgeti(L, LUA_REGISTRYINDEX, v.ref_);
    }
  }
};

template<>
struct Handler<std::nullptr_t> {
  static inline LuaRef Get(lua_State* L, int index, LuaState* s) {
    LOG(FATAL) << "not supported";
    return LuaRef();
  }
  static inline void Push(lua_State* L, const std::nullptr_t& v) {
    lua_pushnil(L);
  }
};

// generic functor to call push the arguments.
struct PushArg {
  lua_State* L;
  template<typename T>
  inline void operator()(const T& v) const {
    Handler<T>::Push(L, v);
  }
};

}  // namespace lua_stack

inline LuaState::LuaState() {
  L_ = luaL_newstate();
  CHECK(L_ != nullptr)
      << "Failed to create new lua state";
  luaL_openlibs(L_);
}

inline LuaState::~LuaState() {
  if (option_ != kThreadLocal && L_ != nullptr) {
    // never close threadlocal, for save destruction.
    lua_close(L_);
  }
}

inline LuaState* LuaState::Create_(Option opt) {
  LuaState* s = new LuaState();
  s->option_ = opt;
  CHECK_NE(opt, kThreadLocal)
      << "use LuaState::ThreadLocalState() to get the thread local state";
  return s;
}

inline void LuaRef::SetByPopStack_(LuaState* s) {
  CHECK(state_ == nullptr);
  lua_State* L = s->L_;
  if (!lua_isnil(L, -1)) {
    ref_ = lua_ref(L, LUA_REGISTRYINDEX);
    state_ = s;
  } else {
    lua_pop(L, 1);
  }
}

// RAII guard to reset stack
struct LuaState::StackReset {
  lua_State* L;
  int top;
  ~StackReset() {
    lua_settop(L, top);
  }
};

template<typename F>
inline void LuaState::PRun_(F f) {
  if (option_ != kLocking) {
    StackReset reset{L_, lua_gettop(L_)};
    if (option_ == kThreadLocal) {
      CHECK_EQ(ThreadLocalState(), this)
          << "Invoke lua from a different thread in ThreadLocal mode.";
    }
    f(L_);
    CHECK_EQ(reset.top, lua_gettop(L_));
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    StackReset reset{L_, lua_gettop(L_)};
    f(L_);
    CHECK_EQ(reset.top, lua_gettop(L_));
  }
}

inline LuaState* LuaState::ThreadLocalState() {
  return ThreadLocalStore<LuaState>::Get();
}

inline LuaRef LuaState::Eval(const char* lua_code) {
  LuaRef ret;
  this->PRun_([this, lua_code, &ret](lua_State* L) {
      luaL_loadstring(L, lua_code);
      CHECK_EQ(lua_pcall(L, 0, 1, 0), 0)
          << "Lua call error: " << lua_tostring(L, -1) << '\n'
          << "---------\n"
          << lua_code
          << "\n----------";
      ret.SetByPopStack_(this);
    });
  return ret;
}

template<typename T>
inline LuaRef LuaState::Convert(const T& value) {
  LuaRef ret;
  this->PRun_([this, &value, &ret](lua_State* L) {
      lua_stack::Handler<T>::Push(L, value);
      ret.SetByPopStack_(this);
    });
  return ret;
}

inline LuaRef LuaState::operator[](const std::string& key) {
  LuaRef ret;
  this->PRun_([this, &key, &ret](lua_State* L) {
      lua_getglobal(L, key.c_str());
      ret.SetByPopStack_(this);
    });
  return ret;
}

inline void LuaState::SetGlobalField(
    const std::string& key, const LuaRef& value) {
  this->PRun_([this, &key, &value](lua_State* L) {
      lua_rawgeti(L, LUA_REGISTRYINDEX, value.ref_);
      lua_setglobal(L, key.c_str());
    });
}

inline LuaRef::LuaRef(const LuaRef& other) {
  if (other.state_ != nullptr) {
    state_ = other.state_;
    state_->PRun_([this, &other](lua_State* L) {
        lua_rawgeti(L, LUA_REGISTRYINDEX, other.ref_);
        ref_ = luaL_ref(L, LUA_REGISTRYINDEX);
      });
  }
}

inline LuaRef::LuaRef(LuaRef&& other) {
  ref_ = other.ref_;
  state_ = other.state_;
  other.state_ = nullptr;
}

inline LuaRef& LuaRef::operator=(LuaRef&& other) {
  LuaRef(std::move(other)).swap(*this);
  return *this;
}

inline LuaRef& LuaRef::operator=(const LuaRef& other) {
  LuaRef(other).swap(*this);
  return *this;
}

inline void LuaRef::swap(LuaRef& other) { // NOLINT(*)
  std::swap(state_, other.state_);
  std::swap(ref_, other.ref_);
}

inline LuaRef::~LuaRef() {
  if (state_ != nullptr) {
    state_->PRun_([this](lua_State* L) {
        luaL_unref(L, LUA_REGISTRYINDEX, ref_);
      });
  }
}

inline bool LuaRef::is_nil() const {
  return state_ == nullptr;
}

std::ostream &operator<<(std::ostream &os, const LuaRef &r) {
  if (!r.is_nil()) {
    r.state_->PRun_([&os, &r](lua_State* L) {
        lua_rawgeti(L, LUA_REGISTRYINDEX, r.ref_);
        int type = lua_type(L, -1);
        switch (type) {
          case LUA_TSTRING:
            os << "lua_string:'" << lua_tostring(L, -1) << "'"; break;
          case LUA_TBOOLEAN:
            os << "lua_bool:" << (lua_toboolean(L, -1) ? "true" : "false"); break;
          case LUA_TNUMBER:
            os << "lua_number:" << lua_tonumber(L, -1); break;
          default:
            os << "lua[ref=" << r.ref_ << ']' << lua_typename(L, type); break;
        }
        lua_pop(L, 1);
      });
  } else {
    os << "lua_nil";
  }
  return os;
}

template<typename T>
inline T LuaRef::Get() const {
  CHECK(state_ != nullptr) << "Get:: LuaRef is nil";
  T ret;
  state_->PRun_([&ret, this](lua_State* L) {
      lua_rawgeti(L, LUA_REGISTRYINDEX, ref_);
      ret = lua_stack::Handler<T>::Get(L, -1, state_);
      lua_pop(L, 1);
    });
  return ret;
}

template<typename T>
inline T* LuaRef::GetUDataPtr() const {
  CHECK(state_ != nullptr) << "Get:: LuaRef is nil";
  T* ret;
  state_->PRun_([&ret, this](lua_State* L) {
      lua_rawgeti(L, LUA_REGISTRYINDEX, ref_);
      ret = reinterpret_cast<T*>(lua_touserdata(L, -1));
      lua_pop(L, 1);
    });
  return ret;
}

// helper function to dispatch varg foreach
template<bool stop, std::size_t I, typename F, typename ...Args>
struct for_each_dispatcher_ {
  static inline void run(const std::tuple<Args...>& args, F f) {
    f(std::get<I>(args));
    for_each_dispatcher_<(I + 1) == sizeof...(Args), (I+1), F, Args...>::run(args, f);
  }
};
// helper function to run foreach
template<std::size_t I, typename F, typename ...Args>
struct for_each_dispatcher_<true, I, F, Args...>  {
  static inline void run(const std::tuple<Args...>& args, F f) {
  }
};

// template function to iterate over tuples
template<typename F, typename ...Args>
inline void for_each(const std::tuple<Args...>& args, F f) {
  for_each_dispatcher_<sizeof...(Args) == 0, 0, F, Args...>::run(args, f);
}

template<typename... Args>
inline LuaRef LuaRef::operator()(Args&& ...args) const {
  CHECK(state_ != nullptr) << "LuaRef is nil";
  auto targ = std::make_tuple(std::forward<Args>(args)...);
  size_t nargs = sizeof...(Args);
  LuaRef ret;
  state_->PRun_([this, nargs, &targ, &ret](lua_State* L) {
      lua_rawgeti(L, LUA_REGISTRYINDEX, this->ref_);
      CHECK(lua_isfunction(L, -1))
          << "Expect to invoke a function but type='"
          << lua_typename(L, lua_type(L, -1)) << '\'';
      for_each(targ, lua_stack::PushArg{L});
      LUA_CALL(lua_pcall(L, nargs, 1, 0));
      ret.SetByPopStack_(state_);
    });
  return ret;
}

template<typename T>
inline LuaRef& LuaRef::SetField(const std::string& key, const T& value) {  // NOLINT(*)
  CHECK(state_ != nullptr) << "LuaRef is nil";
  state_->PRun_([this, &key, &value](lua_State* L) {
      lua_rawgeti(L, LUA_REGISTRYINDEX, this->ref_);
      CHECK(lua_istable(L, -1))
          << "Expect a table but type='"
          << lua_typename(L, lua_type(L, -1)) << '\'';
      lua_stack::Handler<T>::Push(L, value);
      lua_setfield(L, -2, key.c_str());
      lua_pop(L, 1);
    });
  return *this;
}

inline LuaRef LuaRef::operator[](const std::string& key) const {
  CHECK(state_ != nullptr) << "LuaRef is nil";
  LuaRef ret;
  state_->PRun_([this, &key, &ret](lua_State* L) {
      lua_rawgeti(L, LUA_REGISTRYINDEX, this->ref_);
      CHECK(lua_istable(L, -1))
          << "Expect a table but type='"
          << lua_typename(L, lua_type(L, -1)) << '\'';
      lua_getfield(L, -1, key.c_str());
      ret.SetByPopStack_(state_);
      lua_pop(L, 1);
    });
  return ret;
}

inline LuaRef LuaRef::operator[](size_t index) const {
  CHECK(state_ != nullptr) << "LuaRef is nil";
  LuaRef ret;
  state_->PRun_([this, index, &ret](lua_State* L) {
      lua_rawgeti(L, LUA_REGISTRYINDEX, this->ref_);
      CHECK(lua_istable(L, -1))
          << "Expect a table but type='"
          << lua_typename(L, lua_type(L, -1)) << '\'';
      lua_rawgeti(L, -1, index);
      ret.SetByPopStack_(state_);
      lua_pop(L, 1);
    });
  return ret;
}

//! \endcond
}  // namespace dmlc

#endif  // DMLC_LUA_H_
